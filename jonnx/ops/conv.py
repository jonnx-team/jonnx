"""ONNX Conv ops."""
from typing import Sequence, Tuple

from jax import lax
from jonnx.core import node
from jonnx.utils import registry


@registry.register_op('Conv')
class Conv(node.Node):
  """Conv class."""

  @classmethod
  def convert_onnx_pad_to_jax_pad(cls, pads) -> Sequence[Tuple[int, int]]:
    """ONNX Pads convention is [x1_begin, x2_begin,..., x1_end, x2_end,...].

    While JAX pads convention is ([(x1_begin, x1_end), (x2_begin, x2_end),..]

    Args:
      pads: ONNX conversion pads.

    Returns:
      result: JAX convention pads.
    """
    if not pads:
      return pads
    assert isinstance(pads, Sequence)
    length = len(pads)
    assert length % 2 == 0
    print('pads = %s', pads)
    result = []
    for i in range(length // 2):
      result.append((pads[i], pads[i + length // 2]))
    print('result = %s', result)
    return result

  def __call__(self, x, w, b=0):
    group = self.attribute.get('group', 1)
    assert group == 1
    kernel_shape = self.attribute.get('kernel_shape', None)
    kernel_shape = kernel_shape or w.shape
    strides = self.attribute.get('strides', None)
    strides = strides or [1] * (w.ndim - 2)
    auto_pad = self.attribute.get('auto_pad', None)
    pads = self.attribute.get('pads', None)
    pads = self.convert_onnx_pad_to_jax_pad(pads)
    dilations = self.attribute.get('dilations', None)
    if auto_pad and auto_pad != 'NOTSET':
      auto_pad = 'SAME' if auto_pad.startswith(b'SAME') else 'VALID'
      pads = lax.padtype_to_pads(x.shape[2:], w.shape[2:], strides, auto_pad)
    else:
      pads = pads or [0] * (w.ndim - 2)
    lhs_dilation = [1] * (w.ndim - 2)
    rhs_dilation = dilations or [1] * (w.ndim - 2)
    return [
        lax.conv_with_general_padding(x, w, strides, pads, lhs_dilation,
                                      rhs_dilation) + b
    ]
