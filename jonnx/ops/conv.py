"""ONNX Conv ops."""
from jax import lax
from jonnx.utils import registry


@registry.register_op('Conv')
def onnx_conv(x,
              w,
              b=0,
              group=1,
              kernel_shape=None,
              pads=None,
              strides=None,
              dilations=None,
              auto_pad=None,
              **kwargs):
  """Numpy-backed implementation of ONNX Conv op."""
  assert group == 1
  kernel_shape = kernel_shape or w.shape
  strides = strides or [1] * (w.ndim - 2)
  if auto_pad:
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
