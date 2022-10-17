"""ONNX Conv ops."""
from functools import partial
from typing import Sequence, Tuple

import jax
from jax import numpy as jnp
from jax import lax
from jonnx.core import module
from jonnx.core import node
from jonnx.utils import registry
from jonnx.utils.ops_utils import convert_onnx_pad_to_jax_pad


def pad_helper(input_rank, pads=None):
  pad_pairs = len(pads) // 2 if pads else 0

  pad_width = []
  for _ in range(input_rank - pad_pairs):
    pad_width.append((0, 0))
  for idx in range(pad_pairs):
    pad_width.append((pads[idx], pads[idx + pad_pairs]))
  return pad_width


@registry.register_op('MaxPool')
class MaxPool(node.Node):

  @partial(jax.jit, static_argnames={'self'})
  def __call__(self, x):
    pads = self.attribute.get('pads', None)
    auto_pad = self.attribute.get('auto_pad', 'NOTSET')

    if auto_pad == 'NOTSET':
      pads = pad_helper(x.ndim, pads) if pads else 'VALID'
    elif auto_pad == 'SAME_UPPER':
      pads = 'SAME'
    elif auto_pad == 'VALID':
      pads = 'VALID'
    elif auto_pad == 'SAME_LOWER':
      raise NotImplemented('MaxPool with auto_pad `SAME_LOWER`')
    else:
      raise ValueError(f'Invalid auto_pad attribute: {auto_pad}')

    kernel_shape = self.attribute.get('kernel_shape', [])
    prefix = (1,) * (x.ndim - len(kernel_shape))
    dims = prefix + tuple(kernel_shape)
    strides = self.attribute.get('strides', None)
    strides = (prefix + tuple(strides)) if strides else [1] * len(dims)
    dilations = self.attribute.get('dilations', None)
    dilations = (((1,) * (x.ndim - len(dilations)) +
                  tuple(dilations)) if dilations else (1,) * x.ndim)
    return [
        lax.reduce_window(x, -jnp.inf, lax.max, dims, strides, pads, None,
                          dilations)
    ]

  def __post_init__(self):
    super().__post_init__()
    if 'ceil_mode' in self.attribute and self.attribute['ceil_mode']:
      raise ValueError(
          'Currently does not support MaxPool option ceil_mode=True.')
