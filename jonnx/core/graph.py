"""Define Graph class."""
from typing import Dict, Text, List

from onnx import GraphProto


class Graph:
  """Graph class wrapper of ONNX.GraphProto."""

  def __init__(self, graph_proto: GraphProto):
    self.graph = graph_proto

    # Build the ref_dict
    self.ref_dict = {}
    for node in self.graph.node:
      inputs = node.input
      for input_ in inputs:
        if input_ in self.ref_dict:
          self.ref_dict[input_] += 1
        else:
          self.ref_dict[input_] = 1

    # Initialze the tensor_dict
    self.tensor_dict = {}
    for n in self.graph.initializer:
      self.tensor_dict[n.name] = n

    # Build the node_dict
    self.node_dict = {}
    for node in self.graph.node:
      self.node_dict[node.name] = node

    # Build the tensor_to_node_dict
    self.tensor_down_to_node_dict: Dict[Text, List[Text]] = {}
    self.tensor_up_to_node_dict: Dict[Text, Text] = {}
    for node in self.graph.node:

      inputs = node.input
      for input_name in inputs:
        if input_name not in self.tensor_down_to_node_dict:
          self.tensor_down_to_node_dict[input_name] = []
        self.tensor_down_to_node_dict[input_name].append(node.name)

      outputs = node.output
      for output_name in outputs:
        self.tensor_up_to_node_dict[output_name] = node.name

    # Build the node_to_tensor_dict
    self.node_down_to_tensor_dict: Dict[Text, List[Text]] = {}
    self.node_up_to_tensor_dict: Dict[Text, List[Text]] = {}
    for node in self.graph.node:
      outputs_name = [o for o in node.output]
      self.node_down_to_tensor_dict[node.name] = outputs_name
      inputs_name = [i for i in node.input]
      self.node_up_to_tensor_dict[node.name] = inputs_name

  def get_parent_nodes_name(self, node_name: Text) -> List[Text]:
    inputs = self.node_up_to_tensor_dict[node_name]
    results = []
    for input_ in inputs:
      results.append(self.tensor_up_to_node_dict[input_])
    return results

  def get_child_nodes_name(self, node_name: Text) -> List[Text]:
    outputs = self.node_down_to_tensor_dict[node_name]
    results = []
    for output_ in outputs:
      results.extend(self.tensor_down_to_node_dict[output_])
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
