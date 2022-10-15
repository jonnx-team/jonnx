"""define Tensor class."""
from jonnx.core import module
import numpy as np
import onnx
from onnx import TensorProto


class Tensor(module.Module):
  name: str
  value: np.ndarray

  @classmethod
  def from_proto(cls, tensor_proto: TensorProto):
    name = tensor_proto.name
    value = onnx.numpy_helper.to_array(tensor_proto)
    return cls(name, value)

