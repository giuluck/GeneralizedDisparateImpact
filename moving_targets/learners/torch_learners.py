from typing import Optional, List, Union, Tuple, Dict, Any

import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from moving_targets.learners import Learner
from moving_targets.util.scalers import Scaler


class TorchMLP(Learner):
    """Torch Dense Neural Network Wrapper"""

    class Dataset(Dataset):
        """Inner utility class to deal with torch expected input."""

        def __init__(self, x, y):
            import torch
            assert len(x) == len(y), f"Data should have the same length, but len(x) = {len(x)} and len(y) = {len(y)}"
            self.x = torch.tensor(np.array(x), dtype=torch.float32)
            self.y = torch.tensor(np.expand_dims(y, axis=-1), dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    def __init__(self,
                 loss: nn.Module,
                 activation: Optional[nn.Module] = None,
                 hidden_units: List[int] = (128,),
                 optimizer: str = 'Adam',
                 epochs: int = 1,
                 validation_split: float = 0.0,
                 batch_size: Optional[int] = 128,
                 verbose: bool = True,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False):
        """
        :param loss:
            The neural network loss function.

        :param activation:
            The neural network output activation module.

        :param hidden_units:
            The tuple of neural network hidden units.

        :param optimizer:
            The name of the neural network optimizer to retrieve it from `torch.optim`.

        :param epochs:
            The number of training epochs.

        :param validation_split:
            The validation split for neural network training.

        :param batch_size:
            The batch size for neural network training.

        :param verbose:
            Whether or not to print information during the neural network training.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.
        """

        super(TorchMLP, self).__init__(x_scaler=x_scaler, y_scaler=y_scaler, stats=stats)

        self.model: Optional[nn.Module] = None
        """The torch model."""

        self.activation: Optional[nn.Module] = activation
        """The neural network output activation module."""

        self.hidden_units: List[int] = hidden_units
        """The tuple of neural network hidden units."""

        self.loss: nn.Module = loss
        """The neural network loss function."""

        self.optimizer: Any = optimizer
        """The neural network optimizer."""

        self.epochs: int = epochs
        """The number of training epochs."""

        self.validation_split: float = validation_split
        """The validation split for neural network training."""

        self.batch_size: int = batch_size
        """The batch size for neural network training."""

        self.verbose: bool = verbose
        """Whether or not to print information during the neural network training."""

    def _print(self, epoch: int, info: Dict[str, float], batch: Optional[Tuple[int, int]] = None):
        if not self.verbose:
            return

        # carriage return to write in the same line
        print(f'\r', end='')
        # print the epoch number with trailing spaces on the left to match the maximum epoch
        print(f'Epoch {epoch:{len(str(self.epochs))}} -', end='')
        # print the loss value (either loss for this single batch or for the whole epoch)
        for name, value in info.items():
            print(f' {name} = {value:.4f}', end='')
        # check whether this is the information of a single batch or of the whole epoch and for the latter case (batch
        # is None) print a new line, while for the former (batch is a tuple) print the batch number and no new line
        if batch is None:
            print()
        else:
            batch, batches = batch
            print(f' (batch {batch:{len(str(batches))}} of {batches})', end='')

    def _build_model(self, input_units: int) -> optim.Optimizer:
        # build the linear layers and optionally append an output activation layer if passed
        layers = [nn.Linear(in_features=input_units, out_features=self.hidden_units[0]), nn.ReLU()]
        for i in range(1, len(self.hidden_units)):
            layers += [nn.Linear(in_features=self.hidden_units[i - 1], out_features=self.hidden_units[i]), nn.ReLU()]
        layers += [nn.Linear(in_features=self.hidden_units[-1], out_features=1)]
        layers += [] if self.activation is None else [self.activation]
        self.model = nn.Sequential(*layers)
        # build the optimizer
        optimizer = getattr(optim, self.optimizer)
        return optimizer(self.model.parameters())

    def _fit(self, x, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        if sample_weight is not None:
            self.logger.warning("TorchLearner does not support sample weights, please pass 'sample_weight'=None")
        ds = TorchMLP.Dataset(x=x, y=y)
        loader = DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        batches = np.ceil(len(ds) / self.batch_size).astype(int)
        # build the model in order to reinitialize the weights after each iteration
        optimizer = self._build_model(input_units=x.shape[1])
        self.model.train()
        for epoch in np.arange(self.epochs) + 1:
            epoch_loss = 0.0
            for batch, (inp, out) in enumerate(loader):
                optimizer.zero_grad()
                pred = self.model(inp)
                loss = self.loss(pred, out)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inp.size(0)
                self._print(epoch=epoch, info={'loss': loss.item()}, batch=(batch + 1, batches))
            self._print(epoch=epoch, info={'loss': epoch_loss / len(ds)})
        self.model.eval()

    def _predict(self, x) -> np.ndarray:
        import torch
        return self.model(torch.tensor(np.array(x), dtype=torch.float32)).detach().numpy().squeeze()
