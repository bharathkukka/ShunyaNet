# Batch Normalization (NumPy)
# Supports 1D (N, D) for fully-connected layers and 2D/Conv-style (N, C, H, W)
# Implements forward/backward, running statistics, and train/eval modes.
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

Array = np.ndarray


def _validate_array(x: Array, expected_dims: Tuple[int, ...]):
    if x.ndim not in expected_dims:
        raise ValueError(f"Expected input with dims in {expected_dims}, got shape {x.shape} (ndim={x.ndim})")


class BatchNormBase:
    """
    Base class for Batch Normalization.

    Contract:
    - forward(x, training=True) -> y
    - backward(dy) -> dx
    - exposes parameters gamma, beta and their grads dgamma, dbeta when affine=True
    - maintains running_mean, running_var when track_running_stats=True
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.9,
        affine: bool = True,
        track_running_stats: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        if seed is not None:
            np.random.seed(seed)

        # Parameters
        self.gamma: Optional[Array] = np.ones(self.param_shape(), dtype=np.float64) if self.affine else None
        self.beta: Optional[Array] = np.zeros(self.param_shape(), dtype=np.float64) if self.affine else None

        # Running statistics
        self.running_mean: Optional[Array] = np.zeros(self.stat_shape(), dtype=np.float64) if self.track_running_stats else None
        self.running_var: Optional[Array] = np.ones(self.stat_shape(), dtype=np.float64) if self.track_running_stats else None

        # Gradients (populated after backward)
        self.dgamma: Optional[Array] = None
        self.dbeta: Optional[Array] = None

        # Cache for backward
        self._cache = None
        self._training = True

    # Methods that subclasses must implement to define shapes and reduction axes
    def param_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def stat_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def _axes_to_reduce(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def _broadcast_shape(self, x: Array) -> Tuple[int, ...]:
        """Return shape for broadcasting gamma/beta to x."""
        raise NotImplementedError

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def forward(self, x: Array, training: Optional[bool] = None) -> Array:
        """
        Forward pass of BatchNorm.
        x: input array.
        training: override internal mode. If None, use self._training.
        """
        if training is None:
            training = self._training

        # Compute per-feature mean and var over intended axes
        axes = self._axes_to_reduce()
        x_dtype = np.result_type(x, np.float64)
        x = x.astype(x_dtype, copy=False)

        batch_mean = np.mean(x, axis=axes, keepdims=True)
        batch_var = np.var(x, axis=axes, keepdims=True)

        if self.track_running_stats and training:
            # Update running stats (no keepdims for storage)
            mean_no_keep = np.squeeze(batch_mean, axis=axes)
            var_no_keep = np.squeeze(batch_var, axis=axes)
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean_no_keep
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var_no_keep

        if training:
            mean = batch_mean
            var = batch_var
        else:
            if not self.track_running_stats or self.running_mean is None or self.running_var is None:
                # Fall back to batch stats at inference if not tracking
                mean = batch_mean
                var = batch_var
            else:
                # Use running stats; reshape with keepdims to broadcast
                mean = self.running_mean.reshape(self._broadcast_shape(x))
                var = self.running_var.reshape(self._broadcast_shape(x))

        x_centered = x - mean
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = x_centered * inv_std

        if self.affine:
            gamma = self.gamma.reshape(self._broadcast_shape(x))
            beta = self.beta.reshape(self._broadcast_shape(x))
            out = gamma * x_hat + beta
        else:
            out = x_hat

        # Cache for backward only in training mode
        if training:
            self._cache = {
                'x_hat': x_hat,
                'inv_std': inv_std,
                'x_centered': x_centered,
                'axes': axes,
                'shape': x.shape,
            }
        else:
            self._cache = None

        return out

    def backward(self, dout: Array) -> Array:
        if self._cache is None:
            raise RuntimeError("No cache available. Ensure forward(training=True) is called before backward.")
        x_hat = self._cache['x_hat']
        inv_std = self._cache['inv_std']
        x_centered = self._cache['x_centered']
        axes = self._cache['axes']
        N = 1
        for ax in axes:
            N *= x_hat.shape[ax]

        # Ensure dtypes
        dout = dout.astype(x_hat.dtype, copy=False)

        if self.affine:
            # Gradients for gamma and beta
            # Sum over reduction axes to get per-feature grads
            reduce_axes = axes
            self.dgamma = np.sum(dout * x_hat, axis=reduce_axes, keepdims=False)
            self.dbeta = np.sum(dout, axis=reduce_axes, keepdims=False)
            gamma = self.gamma.reshape(self._broadcast_shape(x_hat))
            dxhat = dout * gamma
        else:
            self.dgamma = None
            self.dbeta = None
            dxhat = dout

        # Backprop through normalization
        # Reference formula for BN backward (per feature):
        # dx = (1/N) * inv_std * (N*dxhat - sum(dxhat) - x_hat*sum(dxhat*x_hat))
        sum_dxhat = np.sum(dxhat, axis=axes, keepdims=True)
        sum_dxhat_xhat = np.sum(dxhat * x_hat, axis=axes, keepdims=True)
        dx = (1.0 / N) * inv_std * (N * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
        return dx


class BatchNorm1d(BatchNormBase):
    """Batch Normalization for 2D inputs of shape (N, D)."""

    def param_shape(self) -> Tuple[int, ...]:
        return (self.num_features,)

    def stat_shape(self) -> Tuple[int, ...]:
        return (self.num_features,)

    def _axes_to_reduce(self) -> Tuple[int, ...]:
        # Reduce over batch dimension only
        return (0,)

    def _broadcast_shape(self, x: Array) -> Tuple[int, ...]:
        # gamma/beta of shape (D,) should broadcast to (1, D)
        return (1, self.num_features)


class BatchNorm2d(BatchNormBase):
    """Batch Normalization for conv feature maps of shape (N, C, H, W)."""

    def param_shape(self) -> Tuple[int, ...]:
        return (self.num_features,)

    def stat_shape(self) -> Tuple[int, ...]:
        return (self.num_features,)

    def _axes_to_reduce(self) -> Tuple[int, ...]:
        # Reduce over N, H, W (per-channel statistics)
        return (0, 2, 3)

    def _broadcast_shape(self, x: Array) -> Tuple[int, ...]:
        # gamma/beta of shape (C,) should broadcast to (1, C, 1, 1)
        return (1, self.num_features, 1, 1)


# Functional helpers

def batch_norm_1d_forward(x: Array, gamma: Array, beta: Array, eps: float = 1e-5) -> Tuple[Array, dict]:
    _validate_array(x, (2,))
    mu = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv_std
    out = gamma * x_hat + beta
    cache = {'x_hat': x_hat, 'inv_std': inv_std}
    return out, cache


def batch_norm_1d_backward(dout: Array, cache: dict, gamma: Array) -> Array:
    x_hat = cache['x_hat']
    inv_std = cache['inv_std']
    N = x_hat.shape[0]
    dxhat = dout * gamma
    sum_dxhat = np.sum(dxhat, axis=0, keepdims=True)
    sum_dxhat_xhat = np.sum(dxhat * x_hat, axis=0, keepdims=True)
    dx = (1.0 / N) * inv_std * (N * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    return dx


__all__ = [
    'BatchNorm1d',
    'BatchNorm2d',
    'batch_norm_1d_forward',
    'batch_norm_1d_backward',
]

