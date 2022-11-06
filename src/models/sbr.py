from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.metrics import Mean
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.optimizer_v2.adam import Adam

from src.models import Model


class KerasSBR(KerasModel):
    def __init__(self,
                 data: pd.DataFrame,
                 protected: Union[str, List[str]],
                 classification: bool,
                 threshold: float = 0.0,
                 units: List[int] = (),
                 alpha: Optional[float] = None):
        """
        :param data:
            The input data on which to compute the indicator matrix.

        :param protected:
            The protected features.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param threshold:
            The DIDI threshold.

        :param units:
            The neural network hidden units.

        :param alpha:
            Either a fixed alpha or None to use the lagrangian dual alpha optimization process.
        """
        super(KerasSBR, self).__init__()

        self.threshold: float = threshold
        """The DIDI threshold."""

        protected = protected if isinstance(protected, list) else [protected]
        self.protected: List[int] = [i for i, c in enumerate(data.columns) if c in protected]
        """The indices of the protected features."""

        activation = 'sigmoid' if classification else None
        self.lrs: List[Dense] = [Dense(u, activation='relu') for u in units] + [Dense(1, activation=activation)]
        """The list of neural layers."""

        self.alpha = tf.Variable(0., name='alpha') if alpha is None else tf.Variable(alpha, name='alpha')
        """The alpha value for balancing compiled and regularized loss."""

        self.alpha_optimizer = Adam(learning_rate=1.0) if alpha is None else None
        """The optimizer of the alpha value that leverages the lagrangian dual technique."""

        self._alpha_tracker = Mean(name='alpha')
        """The tracker of alpha values during the training process."""

        self._tot_loss_tracker = Mean(name='tot_loss')
        """The tracker of total loss values during the training process."""

        self._def_loss_tracker = Mean(name='def_loss')
        """The tracker of default loss values during the training process."""

        self._reg_loss_tracker = Mean(name='reg_loss')
        """The tracker of regularizer loss values during the training process."""

        self._perc_didi_tracker = Mean(name='perc_didi')
        """The tracker of percentage didi metric during the training process."""

        self._test_loss_tracker = Mean(name='test_loss')
        """The tracker of test loss values during the training process."""

        # build the model with the correct input shape
        tensor = tf.zeros((1, data.shape[1]))
        self.call(tensor)

    def _absolute_didi(self, x: tf.Tensor, y: tf.Tensor) -> float:
        """Computes the absolute didi given certain input and target tensors.

        :param x:
            The input tensor.

        :param y:
            The output targets.

        :return:
            The absolute didi.
        """
        didi = 0
        avg = tf.reduce_mean(y)
        for protected in self.protected:
            group_mask = x[:, protected]
            group_avg = tf.reduce_mean(y[group_mask == 1.0])
            didi += tf.abs(avg - group_avg)
        return didi

    def _custom_loss(self, x: tf.Tensor, y: tf.Tensor, sign: int = 1) -> Tuple[float, float, float, float]:
        """Computes the custom losses.

        :param x:
            The input data.

        :param y:
            The target data.

        :param sign:
            Whether to minimize the loss (1) or to maximize it (-1) depending on the training step.

        :return:
            A tuple of the form (<total_loss>, <default_loss>, <regularizer_loss>, <didi>), where the total loss is
            computed as: <total_loss> = <sign> * (<default_loss> + <alpha> * <regularizer_loss>)
        """
        # obtain the predictions
        p = self(x, training=True)
        # compute the default loss
        def_loss = self.compiled_loss(y, p)
        # compute the regularization loss
        perc_didi = self._absolute_didi(x, p) / self._absolute_didi(x, y)
        reg_loss = tf.maximum(0.0, perc_didi - self.threshold)
        return sign * (def_loss + self.alpha * reg_loss), def_loss, reg_loss, perc_didi

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """Overrides keras `train_step` method.

        :param data:
            The (x, y) data batch.

        :return:
            A dictionary containing the values of <alpha>, <total_loss>, <default_loss>, and <regularization_loss>.
        """
        # unpack training data
        x, y = data
        nn_vars = self.trainable_variables[:-1]
        alpha_var = self.trainable_variables[-1:]
        # first optimization step: network parameters with alpha (last var) excluded -> loss minimization
        with tf.GradientTape() as tape:
            tot_loss, def_loss, reg_loss, perc_didi = self._custom_loss(x, y, sign=1)
            grads = tape.gradient(tot_loss, nn_vars)
            self.optimizer.apply_gradients(zip(grads, nn_vars))
        # second optimization step: alpha only -> loss maximization
        if self.alpha_optimizer is not None:
            with tf.GradientTape() as tape:
                tot_loss, def_loss, reg_loss, perc_didi = self._custom_loss(x, y, sign=-1)
                grads = tape.gradient(tot_loss, alpha_var)
                self.alpha_optimizer.apply_gradients(zip(grads, alpha_var))
        # loss tracking
        self._alpha_tracker.update_state(self.alpha)
        self._tot_loss_tracker.update_state(abs(tot_loss))
        self._def_loss_tracker.update_state(def_loss)
        self._reg_loss_tracker.update_state(reg_loss)
        self._perc_didi_tracker.update_state(perc_didi)
        return {
            'alpha': self._alpha_tracker.result(),
            'tot_loss': self._tot_loss_tracker.result(),
            'def_loss': self._def_loss_tracker.result(),
            'reg_loss': self._reg_loss_tracker.result(),
            'perc_didi': self._perc_didi_tracker.result()
        }

    def test_step(self, d: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """Overrides keras `test_step` method.

        :param d:
            The batch, having the form (<input_data>, <ground_truths>).

        :return:
            A dictionary containing the values of <test_loss>.
        """
        x, labels = d
        loss = self.compiled_loss(labels, self(x, training=False))
        self._test_loss_tracker.update_state(loss)
        return {
            'loss': self._test_loss_tracker.result()
        }

    def call(self, inputs, training=None, mask=None):
        """Overrides Keras method.

        :param inputs:
            The neural network inputs.

        :param training:
            Overrides Keras parameter.

        :param mask:
            Overrides Keras parameter.
        """
        x = inputs
        for layer in self.lrs:
            x = layer(x)
        return x

    def get_config(self):
        """Overrides Keras method."""
        pass

    def _serialize_to_tensors(self):
        """Overrides Keras method."""
        pass

    def _restore_from_tensors(self, restored_tensors):
        """Overrides Keras method."""
        pass


class SBR(Model):
    def __init__(self,
                 excluded: Union[str, List[str]],
                 classification: bool,
                 threshold: float = 0.0,
                 val_split: float = 0.0,
                 units: List[int] = (128, 128),
                 alpha: Optional[float] = None,
                 epochs: int = 5000,
                 verbose: bool = False,
                 callbacks: List[Callback] = ()):
        """
        :param excluded:
            The features to be excluded.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param threshold:
            The DIDI threshold.

        :param val_split:
            The neural network validation split.

        :param units:
            The neural network hidden units.

        :param alpha:
            Either a fixed alpha or None to use the lagrangian dual alpha optimization process.

        :param epochs:
            The neural network training epochs.

        :param verbose:
            The neural network verbosity.

        :param callbacks:
            The neural network callbacks.
        """
        super(SBR, self).__init__(
            name='sbr',
            classification=classification,
            excluded=excluded,
            threshold=threshold,
            val_split=val_split,
            units=units,
            alpha=alpha,
            epochs=epochs,
            callbacks=callbacks
        )

        self.net: Optional[KerasSBR] = None
        """The neural sbr model."""

        self.net_args: Dict[str, Any] = {
            'alpha': alpha,
            'units': units,
            'protected': excluded,
            'threshold': threshold,
            'classification': classification
        }
        """Custom arguments to be passed to the 'KerasSBR' constructor."""

        self.compile_args: Dict[str, Any] = {
            'loss': 'binary_crossentropy' if classification else 'mse',
            'optimizer': 'adam'
        }
        """Custom arguments to be passed to the 'compile' method."""

        self.fit_args: Dict[str, Any] = {
            'epochs': epochs,
            'verbose': verbose,
            'callbacks': callbacks,
            'validation_split': val_split
        }
        """Custom arguments to be passed to the 'fit' method."""

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        self.net = KerasSBR(data=x, **self.net_args)
        self.net.compile(**self.compile_args)
        self.net.fit(x, y, batch_size=len(x), **self.fit_args)

    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.net.predict(x)
