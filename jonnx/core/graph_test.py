from absl import logging
from absl.testing import absltest
from jonnx.core import graph
from onnx import helper


class GraphTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # https://www.geeksforgeeks.org/topological-sorting/
    # Reconstruct the graph here.
    graph_inputs = ["input"]

    node0 = helper.make_node(
        op_type="Abs",
        inputs=["node5_output", "node4_output"],
        outputs=[],
        name="node0")

    node1 = helper.make_node(
        op_type="Abs",
        inputs=["node3_output", "node4_output"],
        outputs=[],
        name="node1")

    node2 = helper.make_node(
        op_type="Abs",
        inputs=["node5_output"],
        outputs=["node2_output"],
        name="node2")

    node3 = helper.make_node(
        op_type="Abs",
        inputs=["node2_output"],
        outputs=["node3_output"],
        name="node3")

    node4 = helper.make_node(
        op_type="Abs",
        inputs=graph_inputs,
        outputs=["node4_output"],
        name="node4")

    node5 = helper.make_node(
        op_type="Abs",
        inputs=graph_inputs,
        outputs=["node5_output"],
        name="node5")

    self.graph_proto = helper.make_graph(
        nodes=[node0, node1, node2, node3, node4, node5],
        name="graph",
        inputs=[],
        outputs=[],
        initializer=[],
    )

  def test_init(self):
    g = graph.Graph(self.graph_proto)
    logging.info("g node_dict = %s", g.node_dict)
    self.assertIs(len(g.node_dict), 6)

  def test_get_parent_nodes_name(self):
    g = graph.Graph(self.graph_proto)
    parent_nodes = g.get_parent_nodes_name("node1")
    self.assertCountEqual(parent_nodes, ["node4", "node3"])

  def test_get_child_nodes_name(self):
    g = graph.Graph(self.graph_proto)
    child_nodes = g.get_child_nodes_name("node5")
    self.assertCountEqual(child_nodes, ["node0", "node2"])

  def test_topological_sort(self):
    g = graph.Graph(self.graph_proto)
    node_list = g.topological_sort()
    logging.info("node_list = %s", node_list)
    expect_node_list = ["node5", "node4", "node2", "node3", "node1", "node0"]
    self.assertCountEqual(node_list, expect_node_list)


if __name__ == "__main__":
  absltest.main()
