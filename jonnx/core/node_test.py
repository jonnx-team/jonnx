"""Test Module class."""
import dataclasses
from typing import Dict, List
import pprint

from absl.testing import absltest
from jonnx.core import node
import onnx

from jax.tree_util import tree_structure
from jax.tree_util import tree_flatten, tree_unflatten

class NodeTest(absltest.TestCase):
  
  def setUp(self):
    super().setUp()
    self.node_proto = onnx.helper.make_node(
        op_type="Conv",
        inputs=["x", "W"],
        outputs=["y"],
        kernel_shape=[3, 3],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=[1, 1, 1, 1],
    )
    

  def test_basic(self):
    node_obj = node.Node(self.node_proto)
    self.assertEqual(node_obj.op_type, "Conv")
    self.assertEqual(node_obj.name, "")
    
  def test_pytree(self):
    node_obj = node.Node(self.node_proto)    
    value_flat, value_tree = tree_flatten(node_obj)    
    new_node_obj = tree_unflatten(value_tree, value_flat)
    self.assertEqual(node_obj, new_node_obj)

if __name__ == "__main__":
  absltest.main()
