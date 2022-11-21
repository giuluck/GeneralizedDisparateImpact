import logging
from typing import List, Union, Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.metrics import HGR
from src.models.base import NeuralNetwork


class NeuralSBR(NeuralNetwork):
    EPS: float = 1e-6
    """Relative epsilon value used to account for numerical errors in higher-order covariance penalties."""

    def __init__(self,
                 penalty: str,
                 classification: bool,
                 excluded: Union[str, List[str]],
                 threshold: float = 0.0,
                 degrees: int = 1,
                 validation_split: float = 0.0,
                 hidden_units: List[int] = (128, 128),
                 batch_size: int = 128,
                 epochs: int = 200,
                 verbose: bool = False):
        """
        :param penalty:
            The type of penalty to use, either 'cov' or 'hgr'.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param excluded:
            The features to be excluded.

        :param threshold:
            The exclusion threshold.

        :param degrees:
            The kernel degrees used for the features to be excluded (used for constraint = 'cov' only).

        :param validation_split:
            The neural network validation split.

        :param hidden_units:
            The neural network hidden units.

        :param batch_size:
            The neural network batch size.

        :param epochs:
            The neural network training epochs.

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
        if penalty == 'cov':
            reg_vector = self._cov_penalty
        elif penalty == 'hgr':
            if degrees != 1:
                logging.log(level=logging.WARNING, msg=f"HGR does not accept degree > 1, use 1 instead of {degrees}")
                degrees = 1
            reg_vector = self._hgr_penalty
        else:
            raise ValueError(f"Unknown penalty '{penalty}'")

        # update name and configuration
        self.__name__: str = f'sbr {penalty}'
        self.config['penalty'] = penalty
        self.config['degrees'] = degrees
        self.config['excluded'] = excluded
        self.config['threshold'] = threshold

        self.classification: bool = classification
        """Whether we are dealing with a binary classification or a regression task."""

        self.excluded: List[Any] = excluded if isinstance(excluded, list) else [excluded]
        """The features to be excluded."""

        self.threshold: float = threshold
        """The exclusion threshold."""

        self.degrees: int = degrees
        """The kernel degrees used for the features to be excluded."""

        self.alpha: Variable = Variable(torch.zeros(len(self.excluded) * degrees), requires_grad=True, name='alpha')
        """The alpha value for balancing compiled and regularized loss."""

        self.alpha_optimizer: optim.Optimizer = optim.Adam(params=[self.alpha])
        """The optimizer of the alpha value that leverages the lagrangian dual technique."""

        self.regularization_vector: Callable = reg_vector
        """A function f(x, y, p) -> v which returns a vector of partial losses, one per each constraint."""

    def _cov_penalty(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # the total regularization is given by the sum of violations per excluded feature
        # where the violation of each excluded feature is compute as the sum of violations on the covariances
        reg_vector = []
        for feature in self.excluded:
            # here we cannot use the covariance formulation as in Moving Targets since we have no guarantees that the
            # weights will be correctly constrained (thus no assumption on higher-order weights can be done)
            # therefore we will use torch implementation of the least square errors
            # (the 'gelsd' driver allows to have both more precise and more reproducible results)
            z = x[:, feature]
            z = torch.stack([z ** d for d in range(self.degrees + 1)], dim=1)
            wp, _, _, _ = torch.linalg.lstsq(z, p, driver='gelsd')
            wy, _, _, _ = torch.linalg.lstsq(z, y, driver='gelsd')
            # we multiply the threshold by the relative weight in order to avoid numerical errors
            reg_vector += [torch.maximum(torch.zeros(1), torch.abs(wp[1]) - self.threshold * torch.abs(wy[1]))]
            # the violation of the higher orders is computed as the absolute value of the respective weight
            reg_vector += [torch.maximum(torch.zeros(1), torch.abs(wp[2:, 0]) - self.EPS)]
        # finally, we concatenate all the values to obtain a constraint vector
        return torch.concatenate(reg_vector)

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
        y, p = y.squeeze(), p.squeeze()
        reg_vector = []
        for feature in self.excluded:
            # retrieve the correct feature to exclude
            z = x[:, feature]
            # compute the \chi^2 value for the predictions and for the original targets
            chi2_p = HGR.hgr(p, z, chi2=True)
            chi2_y = HGR.hgr(y, z, chi2=True)
            # constraint the relative \chi^2 by multiplying the threshold to avoid numerical errors
            reg_vector += [torch.maximum(torch.zeros(1), chi2_p - self.threshold * chi2_y)]
        # finally, we concatenate all the values to obtain a constraint vector
        return torch.concatenate(reg_vector)

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        # change feature names with feature indices
        self.excluded = [i for i, c in enumerate(x.columns) if c in self.excluded]
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
                reg_vector = self.regularization_vector(inp, out, pred)
                reg_loss = self.alpha @ reg_vector
                tot_loss = def_loss + reg_loss
                tot_loss.backward()
                optimizer.step()
                # alpha maximization step
                self.alpha_optimizer.zero_grad()
                pred = self.model(inp)
                def_loss = self.loss(pred, out)
                reg_vector = self.regularization_vector(inp, out, pred)
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
