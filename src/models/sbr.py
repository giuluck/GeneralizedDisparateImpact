import logging
from typing import List, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.metrics import HGR, GeneralizedDIDI as gDIDI
from src.models.base import NeuralNetwork


class NeuralSBR(NeuralNetwork):
    EPS: float = 1e-6
    """Relative epsilon value used to account for numerical errors in higher-order covariance penalties."""

    def __init__(self,
                 penalty: str,
                 excluded: str,
                 classification: bool,
                 degree: int,
                 threshold: float,
                 relative: Union[bool, int],
                 hidden_units: List[int],
                 batch_size: int,
                 epochs: int,
                 validation_split: float = 0.0,
                 verbose: bool = False):
        """
        :param penalty:
            The type of penalty to use, either 'coarse' (coarse-grained GeDI constraint with a single penalizer),
            'fine' (fine-grained GeDI constraint on first order with multiple penalizers), or 'hgr' (chi^2 constraint
            with a single penalizer).

        :param excluded:
            The feature to be excluded.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param degree:
            The kernel degree for the excluded feature. If the penalty is 'hgr', this value is ignored.

        :param threshold:
            The exclusion threshold.

        :param relative:
            If a positive integer k is passed, it computes the relative value with respect to the indicator computed on
            the original targets with kernel k. If True is passed, it assumes k = 1. Otherwise, if False is passed, it
            simply computes the absolute value of the indicator.

        :param hidden_units:
            The neural network hidden units.

        :param batch_size:
            The neural network batch size.

        :param epochs:
            The neural network training epochs.

        :param validation_split:
            The neural network validation split.

        :param verbose:
            The neural network verbosity.
        """
        super(NeuralSBR, self).__init__(
            classification=classification,
            validation_split=validation_split,
            hidden_units=hidden_units,
            batch_size=batch_size,
            verbose=verbose,
            epochs=epochs,
            logger=None
        )

        # validate constraint and degrees
        assert degree > 0, f"The kernel degree must be a positive integer, got {degree}"
        if penalty == 'coarse':
            reg_vector = self._didi_penalty
            penalizers = 1
        elif penalty == 'fine':
            reg_vector = self._first_penalty
            penalizers = degree
        elif penalty == 'hgr':
            if degree != 1:
                logging.log(level=logging.WARNING, msg=f"HGR does not accept degree > 1, use 1 instead of {degree}")
                degree = 1
            if isinstance(relative, int) and relative > 1:
                logging.log(level=logging.WARNING, msg=f"HGR does not accept relative > 1, use 1 instead of {relative}")
                relative = 1
            penalizers = 1
            reg_vector = self._hgr_penalty
        else:
            raise ValueError(f"Unknown penalty '{penalty}'")

        # update name and configuration
        self.__name__: str = f'sbr {penalty}'
        self.config['penalty'] = penalty
        self.config['degree'] = degree
        self.config['excluded'] = excluded
        self.config['threshold'] = threshold
        self.config['relative'] = relative

        self.classification: bool = classification
        """Whether we are dealing with a binary classification or a regression task."""

        self.excluded: str = excluded
        """The feature to be excluded."""

        self.excluded_index: Optional[int] = None
        """The index of the feature to be excluded."""

        self.threshold: float = threshold
        """The exclusion threshold."""

        self.relative: int = int(relative) if isinstance(relative, bool) else relative
        """The kernel degree to use to compute the metric in relative value, or 0 for absolute value."""

        self.degree: int = degree
        """The kernel degree used for the features to be excluded."""

        self.alpha: Variable = Variable(torch.zeros(penalizers), requires_grad=True, name='alpha')
        """The alpha value for balancing compiled and regularized loss."""

        self.alpha_optimizer: optim.Optimizer = optim.Adam(params=[self.alpha])
        """The optimizer of the alpha value that leverages the lagrangian dual technique."""

        self.regularization_vector: Callable = reg_vector
        """A function f(x, y, p) -> v which returns a vector of partial losses, one per each constraint."""

    def _first_penalty(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # for this penalty, we constraint the relative didi to be lower than the threshold (optionally scaled by the
        # value computed on the original targets with degree k if relative is a positive integer k) and, additionally,
        # we constraint the other weights to be null (smaller than an epsilon) to force the higher-orders exclusion
        didi_p, alpha = gDIDI.generalized_didi(x, p, degree=self.degree, use_torch=True, return_weights=True)
        didi_y = gDIDI.generalized_didi(x, y, degree=self.relative, use_torch=True) if self.relative > 0 else 1
        return torch.concatenate((
            torch.maximum(torch.zeros(1), didi_p / didi_y - self.threshold),
            torch.maximum(torch.zeros(1), torch.abs(alpha[1:]) - self.EPS)
        ))

    def _didi_penalty(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # for this penalty, we simply constraint the relative didi to be lower than the threshold (optionally scaled by
        # the value computed on the original targets with degree k if relative is a positive integer k)
        didi_p = gDIDI.generalized_didi(x, p, degree=self.degree, use_torch=True)
        didi_y = gDIDI.generalized_didi(x, y, degree=self.relative, use_torch=True) if self.relative > 0 else 1
        return torch.maximum(torch.zeros(1), didi_p / didi_y - self.threshold)

    def _hgr_penalty(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # This code has been taken from: https://github.com/criteo-research/continuous-fairness/.
        #
        # It computes the \chi^2 divergence between the joint distribution on (x,y) and the product of marginals.
        # This is know to be the square of an upper-bound on the HGR maximum correlation coefficient.
        # We compute it here on an empirical and discretized density estimated from the input data.
        # It returns a numerical value between 0 and infinity, where 0 means independent.
        #
        # Differently from https://github.com/criteo-research/continuous-fairness/, compute a relative \chi^2 (as well
        # as for the other metrics), i.e., instead of constraining the value \chi^2(p, z), we constraint the value
        # \chi^2(p, z) / \chi^2(y, z). In this way, we should have both more explainable and more comparable results.
        # Moreover, we also deal with the case in which we have more than one feature by adding all the \chi^2.
        chi2_p = HGR.hgr(p, x, chi2=True)
        chi2_y = HGR.hgr(y, x, chi2=True) if self.relative > 0 else 1
        return torch.maximum(torch.zeros(1), chi2_p / chi2_y - self.threshold)

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        # get feature index from feature name
        self.excluded_index = x.columns.get_loc(self.excluded)
        # load data and start the model training
        ds = NeuralNetwork.Dataset(x=np.array(x), y=np.array(y))
        loader = DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        batches = np.ceil(len(ds) / self.batch_size).astype(int)
        optimizer = self._build_model(x.shape[1])
        self.model.train()
        for epoch in np.arange(self.epochs) + 1:
            # default loss, regularization loss, and total loss containers
            epoch_tot, epoch_def, epoch_reg, epoch_vec = 0.0, 0.0, 0.0, torch.zeros_like(self.alpha)
            for batch, (inp, out) in enumerate(loader):
                # loss minimization step
                optimizer.zero_grad()
                pred = self.model(inp)
                def_loss = self.loss(pred, out)
                reg_vector = self.regularization_vector(inp[:, self.excluded_index], out.squeeze(), pred.squeeze())
                reg_loss = self.alpha @ reg_vector
                tot_loss = def_loss + reg_loss
                tot_loss.backward()
                optimizer.step()
                # alpha maximization step
                self.alpha_optimizer.zero_grad()
                pred = self.model(inp)
                def_loss = self.loss(pred, out)
                reg_vector = self.regularization_vector(inp[:, self.excluded_index], out.squeeze(), pred.squeeze())
                reg_loss = self.alpha @ reg_vector
                tot_loss = -def_loss - reg_loss
                tot_loss.backward()
                self.alpha_optimizer.step()
                # update the final losses and print the partial results
                epoch_tot += tot_loss.item() * inp.size(0)
                epoch_def += def_loss.item() * inp.size(0)
                epoch_reg += reg_loss.item() * inp.size(0)
                epoch_vec += reg_vector * inp.size(0)
                self._print(epoch=epoch, batch=(batch + 1, batches), info={
                    'tot_loss': f'{-tot_loss.item():4f}',
                    'def_loss': f'{def_loss.item():4f}',
                    'reg_loss': f'{reg_loss.item():4f}',
                    'reg_vec': reg_vector.detach().numpy(),
                    'alpha': self.alpha.detach().numpy()
                })
            self._print(epoch=epoch, info={
                'tot_loss': f'{-epoch_tot / len(ds):4f}',
                'def_loss': f'{epoch_def / len(ds):4f}',
                'reg_loss': f'{epoch_reg / len(ds):4f}',
                'reg_vec': epoch_vec.detach().numpy() / len(ds),
                'alpha': self.alpha.detach().numpy()
            })
        self.model.eval()
