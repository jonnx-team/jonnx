"""Those simple ONNX ops."""
from jax import lax
from jax import numpy as jnp
from jonnx.utils import registry
import numpy as np


@registry.register_op('Abs')
def onnx_abs(x, **kwargs):
  return [lax.abs(x)]


@registry.register_op('Acos')
def onnx_acos(x, **kwargs):
  return [lax.acos(x)]


@registry.register_op('Acosh')
def onnx_acosh(x, **kwargs):
  return [lax.acosh(x)]


@registry.register_op('Constant')
def onnx_constant(**kwargs):
  output_name = kwargs['__output__'][0]
  assert output_name in kwargs, f'{output_name}, kwargs key = {kwargs.keys()}'
  return [kwargs[output_name]]


@registry.register_op('MatMul')
def onnx_matmul(x, y, **kwargs):
  return [jnp.matmul(x, y)]
