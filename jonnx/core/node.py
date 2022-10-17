"""Wrap NodeProto as Node class."""
from typing import Sequence, Optional, Any, Dict

import jonnx
from jonnx.core import module
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

  @classmethod
  def from_proto(cls, node_proto: NodeProto):
    input_ = list(node_proto.input)
    output_ = list(node_proto.output)
    name_ = str(node_proto.name)
    op_type_ = str(node_proto.op_type)
    domain_ = str(node_proto.domain)
    attribute_ = {
        a.name: jonnx.utils.helper.get_attribute_value(a)
        for a in node_proto.attribute
    }
    doc_string_ = str(node_proto.doc_string)
    return cls(
        input=input_,
        output=output_,
        name=name_,
        op_type=op_type_,
        domain=domain_,
        attribute=attribute_,
        doc_string=doc_string_)

  def __call__(self, *args, **kwargs):
    classname = self.__class__.__name__
    raise RuntimeError(f'{classname} class forget implement __call__.')

  def __post_init__(self):
    if not self.op_type:
      raise ValueError('op_type not set.')
