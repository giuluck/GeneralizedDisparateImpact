from typing import List, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.metrics import HGR
from src.models.base import NeuralNetwork


class NeuralSBR(NeuralNetwork):
    def __init__(self,
                 classification: bool,
                 excluded: Union[str, List[str]],
                 threshold: float = 0.0,
                 validation_split: float = 0.0,
                 hidden_units: List[int] = (128, 128),
                 batch_size: int = 128,
                 epochs: int = 200,
                 verbose: bool = False):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param excluded:
            The features to be excluded.

        :param threshold:
            The exclusion threshold.

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

        self.excluded: List[Any] = excluded if isinstance(excluded, list) else [excluded]
        """The features to be excluded."""

        self.threshold: float = threshold
        """The exclusion threshold."""

        self.alpha = Variable(torch.Tensor([0.]), requires_grad=True, name='alpha')
        """The alpha value for balancing compiled and regularized loss."""

        self.alpha_optimizer = Adam(lr=1e-2, params=[self.alpha])
        """The optimizer of the alpha value that leverages the lagrangian dual technique."""

    def _regularization_loss(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Please implement abstract method '_regularization_loss'.")

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
            epoch_def, epoch_reg, epoch_tot = 0.0, 0.0, 0.0
            for batch, (inp, out) in enumerate(loader):
                # loss minimization step
                optimizer.zero_grad()
                pred = self.model(inp)
                def_loss = self.loss(pred, out)
                reg_loss = self._regularization_loss(inp, out, pred)
                tot_loss = def_loss + self.alpha * reg_loss
                tot_loss.backward()
                optimizer.step()
                # alpha maximization step
                self.alpha_optimizer.zero_grad()
                pred = self.model(inp)
                def_loss = self.loss(pred, out)
                reg_loss = self._regularization_loss(inp, out, pred)
                tot_loss = -(def_loss + self.alpha * reg_loss)
                tot_loss.backward()
                self.alpha_optimizer.step()
                # update the final losses and print the partial results
                epoch_def += def_loss.item() * inp.size(0)
                epoch_reg += reg_loss.item() * inp.size(0)
                epoch_tot += tot_loss.item() * inp.size(0)
                self._print(epoch=epoch, batch=(batch + 1, batches), info={
                    'alpha': self.alpha.item(),
                    'def_loss': def_loss.item(),
                    'reg_loss': reg_loss.item(),
                    'tot_loss': -tot_loss.item()
                })
            self._print(epoch=epoch, info={
                'alpha': self.alpha.item(),
                'def_loss': epoch_def / len(ds),
                'reg_loss': epoch_reg / len(ds),
                'tot_loss': -epoch_tot / len(ds)
            })
        self.model.eval()


class CovarianceSBR(NeuralSBR):
    def __init__(self,
                 classification: bool,
                 excluded: Union[str, List[str]],
                 threshold: float = 0.0,
                 degrees: int = 1,
                 gamma: float = 0.1,
                 validation_split: float = 0.0,
                 hidden_units: List[int] = (128, 128),
                 batch_size: int = 128,
                 epochs: int = 200,
                 verbose: bool = False):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param excluded:
            The features to be excluded.

        :param threshold:
            The exclusion threshold.

        :param degrees:
            The kernel degree used for the features to be excluded.

        :param gamma:
            The weight of the higher orders violations with respect to the first order degree one.

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
        super(CovarianceSBR, self).__init__(
            excluded=excluded,
            threshold=threshold,
            classification=classification,
            validation_split=validation_split,
            hidden_units=hidden_units,
            batch_size=batch_size,
            verbose=verbose,
            epochs=epochs,
        )

        self.degrees: int = degrees
        """The kernel degrees used for the features to be excluded."""

        self.gamma: float = gamma
        """The weight of the higher orders violations with respect to the first order degree one."""

        # update name and configuration
        self.__name__: str = 'sbr cov'
        self.config['degrees'] = degrees
        self.config['gamma'] = gamma

    def _regularization_loss(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # the total regularization is given by the sum of violations per excluded feature
        # where the violation of each excluded feature is compute as the sum of violations on the covariances
        reg_loss = torch.zeros(1)
        for feature in self.excluded:
            # instead of using the covariance formulation as in Moving Targets, here we can use the least squares
            z = x[:, [feature]]
            z = torch.concatenate([z ** d for d in range(self.degrees + 1)], dim=1)
            wp, _, _, _ = torch.linalg.lstsq(z, p)
            wy, _, _, _ = torch.linalg.lstsq(z, y)
            wp = torch.abs(wp)
            # we multiply the threshold by the relative weight in order to avoid numerical errors
            reg_loss += torch.maximum(torch.zeros(1), wp[1] - self.threshold * torch.abs(wy[1]))
            # the violation of the higher orders is computed as the absolute distance from the slope at degree one
            reg_loss += self.gamma * torch.sum(torch.abs(wp[2:] - wp[1]))
        return reg_loss


class HirschfeldGebeleinRenyiSBR(NeuralSBR):
    def __init__(self,
                 classification: bool,
                 excluded: Union[str, List[str]],
                 threshold: float = 0.0,
                 validation_split: float = 0.0,
                 hidden_units: List[int] = (128, 128),
                 batch_size: int = 128,
                 epochs: int = 200,
                 verbose: bool = False):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param excluded:
            The features to be excluded.

        :param threshold:
            The exclusion threshold.

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
        super(HirschfeldGebeleinRenyiSBR, self).__init__(
            excluded=excluded,
            threshold=threshold,
            classification=classification,
            validation_split=validation_split,
            hidden_units=hidden_units,
            batch_size=batch_size,
            verbose=verbose,
            epochs=epochs,
        )

        # update name and configuration
        self.__name__: str = 'sbr hgr'

    def _regularization_loss(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
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
        reg_loss = torch.zeros(1)
        for feature in self.excluded:
            # retrieve the correct feature to exclude
            z = x[:, feature]
            # compute the \chi^2 value for the predictions
            h2d = HGR.joint_2(p, z)
            marginal_p = h2d.sum(dim=1).unsqueeze(1)
            marginal_z = h2d.sum(dim=0).unsqueeze(0)
            q = h2d / (torch.sqrt(marginal_p) * torch.sqrt(marginal_z))
            chi2_p = (q ** 2).sum(dim=[0, 1]) - 1.0
            # compute the \chi^2 value for the original targets
            h2d = HGR.joint_2(y, z)
            marginal_y = h2d.sum(dim=1).unsqueeze(1)
            marginal_z = h2d.sum(dim=0).unsqueeze(0)
            q = h2d / (torch.sqrt(marginal_y) * torch.sqrt(marginal_z))
            chi2_y = (q ** 2).sum(dim=[0, 1]) - 1.0
            # constraint the relative \chi^2 by multiplying the threshold to avoid numerical errors
            reg_loss += torch.maximum(torch.zeros(1), chi2_p - self.threshold * chi2_y)
        return reg_loss
