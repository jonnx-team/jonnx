"""Define Graph class."""
from typing import Any, Sequence, Dict, Text, List, Optional

from jonnx.core import module
from jonnx.core import node
from jonnx.core import tensor
from jonnx.core import valueinfo
from jonnx.utils import registry
from onnx import GraphProto

static_field = module.static_field


class Graph(module.Module):
  """Graph class wrapper of ONNX.GraphProto."""
  node_dict: Dict[str, node.Node]
  initializer_dict: Dict[str, tensor.Tensor]
  name: Optional[str] = static_field()
  doc_string: Optional[str] = static_field()
  input: Sequence[str] = static_field()
  output: Sequence[str] = static_field()
  # Change value_info into dict and include both input and output
  value_info_dict: Dict[str, valueinfo.ValueInfoProto] = static_field()
  # extra info beside ONNX Protobuf
  metadata: Any

  def __init__(self, graph_proto: GraphProto):
    self.node_dict = {}
    for nd in graph_proto.node:
      self.node_dict[nd.name] = registry.op(nd.op_type)(nd)
    self.initializer_dict = {
        ts.name: tensor.Tensor.from_proto()(ts) for ts in graph_proto.initializer
    }
    self.input = [proto.name for proto in graph_proto.input]
    self.output = [proto.name for proto in graph_proto.output]
    self.doc_string = graph_proto.doc_string
    self.name = graph_proto.name
    self.value_info_dict = {
        **{
            proto.name: valueinfo.ValueInfo(proto)
            for proto in graph_proto.input
        },
        **{
            proto.name: valueinfo.ValueInfo(proto)
            for proto in graph_proto.output
        },
        **{
            proto.name: valueinfo.ValueInfo(proto)
            for proto in graph_proto.value_info
        },
    }
    self.metadata = {}
    self.create_tensor_and_node_dict()

  def create_ref_dict(self) -> Dict[str, int]:
    """Create reference counting dict."""
    ref_dict = {}
    for _, nd in self.node_dict.items():
      inputs = nd.input
      for input_ in inputs:
        if input_ in ref_dict:
          ref_dict[input_] += 1
        else:
          ref_dict[input_] = 1
    return ref_dict

  def create_tensor_and_node_dict(self):
    # Build the tensor_to_node_dict
    tensor_down_to_node_dict: Dict[Text, List[Text]] = {}
    tensor_up_to_node_dict: Dict[Text, Text] = {}
    for nd_name, nd in self.node_dict.items():
      inputs = nd.input
      for input_name in inputs:
        if input_name not in tensor_down_to_node_dict:
          tensor_down_to_node_dict[input_name] = []
        tensor_down_to_node_dict[input_name].append(nd_name)

      outputs = nd.output
      for output_name in outputs:
        tensor_up_to_node_dict[output_name] = nd_name

    # Build the node_to_tensor_dict
    node_down_to_tensor_dict: Dict[Text, List[Text]] = {}
    node_up_to_tensor_dict: Dict[Text, List[Text]] = {}
    for nd_name, nd in self.node_dict.items():
      outputs_name = [o for o in nd.output]
      node_down_to_tensor_dict[nd_name] = outputs_name
      inputs_name = [i for i in nd.input]
      node_up_to_tensor_dict[nd_name] = inputs_name

    self.metadata['tensor_down_to_node_dict'] = tensor_down_to_node_dict
    self.metadata['tensor_up_to_node_dict'] = tensor_up_to_node_dict
    self.metadata['node_down_to_tensor_dict'] = node_down_to_tensor_dict
    self.metadata['node_up_to_tensor_dict'] = node_up_to_tensor_dict

  def get_parent_nodes_name(self, node_name: Text) -> List[Text]:
    node_up_to_tensor_dict = self.metadata['node_up_to_tensor_dict']
    tensor_up_to_node_dict = self.metadata['tensor_up_to_node_dict']
    inputs = node_up_to_tensor_dict[node_name]
    results = []
    for input_ in inputs:
      results.append(tensor_up_to_node_dict[input_])
    return results

  def get_child_nodes_name(self, node_name: Text) -> List[Text]:
    node_down_to_tensor_dict = self.metadata['node_down_to_tensor_dict']
    tensor_down_to_node_dict = self.metadata['tensor_down_to_node_dict']
    outputs = node_down_to_tensor_dict[node_name]
    results = []
    for output_ in outputs:
      if output_ in tensor_down_to_node_dict:
        results.extend(tensor_down_to_node_dict[output_])
    return results

  def topological_sort(self):
    """Return the topological sort order of those nodes."""

    visited = {}
    stack = []

    # A recursive function used by topologicalSort
    def topological_sort_util(v):
      visited[v] = True
      for i in self.get_child_nodes_name(v):
        if i not in visited or not visited[i]:
          topological_sort_util(i)
      stack.append(v)

    for i in self.node_dict:
      if i not in visited or not visited[i]:
        topological_sort_util(i)

    # return list in reverse order.
    return stack[::-1]
    
