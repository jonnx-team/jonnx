"""define Tensor class."""
from jonnx.core import module
import numpy as np
import onnx
from onnx import TensorProto

static_field = module.static_field


class Tensor(module.Module):
  name: str = static_field()
  value: np.ndarray

  def __init__(self, tensor_proto: TensorProto):
    self.name = tensor_proto.name
    self.value = onnx.numpy_helper.to_array(tensor_proto)
