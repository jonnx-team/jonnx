"""Wrap NodeProto as Node class."""
from typing import Sequence, Optional, Any, Dict

from jonnx.core import module
from onnx import helper
from onnx import NodeProto

static_field = module.static_field


class Node(module.Module):
  """Node class."""
  input: Sequence[str] = static_field()
  output: Sequence[str] = static_field()
  name: Optional[str] = static_field()
  op_type: Optional[str] = static_field()
  domain: Optional[str] = static_field()
  attribute: Dict[str, Any] = static_field()
  doc_string: Optional[str] = static_field()

  def __init__(self, node_proto: NodeProto):
    self.input = node_proto.input
    self.output = node_proto.output
    self.name = node_proto.name
    self.op_type = node_proto.op_type
    self.domain = node_proto.domain
    self.attribute = {
        a.name: helper.get_attribute_value(a) for a in node_proto.attribute
    }
    self.doc_string = node_proto.doc_string

  def __call__(self, *args, **kwargs):
    classname = self.__class__.__name__
    raise RuntimeError(f"{classname} class forget implement __call__.")
  