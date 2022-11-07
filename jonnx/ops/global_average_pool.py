"""ONNX Pool ops."""
from functools import partial
from typing import Sequence, Tuple

import jax
from jax import numpy as jnp
from jax import lax
from jonnx.core import module
from jonnx.core import node
from jonnx.utils import registry


@registry.register_op('GlobalAveragePool')
class GlobalAveragePool(node.Node):

  def __call__(self, x):
    spatial_dim = jnp.ndim(x) - 2
    y = jnp.mean(x, axis=tuple(range(spatial_dim, spatial_dim + 2)))
    for _ in range(spatial_dim):
      y = jnp.expand_dims(y, -1)
    return y
