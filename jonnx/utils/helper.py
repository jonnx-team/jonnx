"""Enhanced version helper class."""
from typing import Any
from onnx import AttributeProto


def get_attribute_value(attr: AttributeProto) -> Any:
  if attr.type == AttributeProto.FLOAT:
    return attr.f
  if attr.type == AttributeProto.INT:
    return attr.i
  if attr.type == AttributeProto.STRING:
    return str(attr.s, 'utf-8')
  if attr.type == AttributeProto.TENSOR:
    return attr.t
  if attr.type == AttributeProto.SPARSE_TENSOR:
    return attr.sparse_tensor
  if attr.type == AttributeProto.GRAPH:
    return attr.g
  if attr.type == AttributeProto.TYPE_PROTO:
    return attr.tp
  if attr.type == AttributeProto.FLOATS:
    return list(attr.floats)
  if attr.type == AttributeProto.INTS:
    return list(attr.ints)
  if attr.type == AttributeProto.STRINGS:
    str_list = list(attr.strings)
    str_list = tuple(map(lambda x: str(x, 'utf-8'), str_list))
    return str_list
  if attr.type == AttributeProto.TENSORS:
    return list(attr.tensors)
  if attr.type == AttributeProto.SPARSE_TENSORS:
    return list(attr.sparse_tensors)
  if attr.type == AttributeProto.GRAPHS:
    return list(attr.graphs)
  if attr.type == AttributeProto.TYPE_PROTOS:
    return list(attr.type_protos)
  raise ValueError(f'Unsupported ONNX attribute: {attr}')
