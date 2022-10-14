"""define ValueInfo class."""
from typing import Optional

from jonnx.core import module
from onnx import ValueInfoProto

static_field = module.static_field


class ValueInfo(module.Module):
  """Wrap the ValueInfoProto class."""
  name: str = static_field()
  doc_string: Optional[str] = static_field()
  shape: Optional[str] = static_field()
  elem_type: int = static_field()

  def __init__(self, value_info_proto: ValueInfoProto):
    self.name = value_info_proto.name
    self.doc_string = value_info_proto.doc_string
    type_proto = value_info_proto.type
    tensor_type_proto = type_proto.tensor_type
    self.elem_type = tensor_type_proto.elem_type
    tensor_shape_proto = tensor_type_proto.shape
    dims = tensor_shape_proto.dim
    self.shape = [dim.WhichOneof("value") for dim in dims]
