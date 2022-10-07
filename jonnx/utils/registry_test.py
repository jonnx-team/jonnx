"""Tests for registry."""
from absl.testing import absltest
from jonnx.utils import registry


class RegistryClassTest(absltest.TestCase):
  """Test of base registry.Registry class."""

  def testGetterSetter(self):
    r = registry.Registry("test_registry")
    r["hello"] = lambda: "world"
    r["a"] = lambda: "b"
    self.assertEqual(r["hello"](), "world")
    self.assertEqual(r["a"](), "b")

  def testDefaultKeyFn(self):
    r = registry.Registry("test", default_key_fn=lambda x: x().upper())
    r.register()(lambda: "hello")
    self.assertEqual(r["HELLO"](), "hello")

  def testNoKeyProvided(self):
    r = registry.Registry("test")

    def f():
      return 3

    r.register(f)
    self.assertEqual(r["f"](), 3)

  def testMembership(self):
    r = registry.Registry("test_registry")
    r["a"] = lambda: None
    r["b"] = lambda: 4
    self.assertIn("a", r)
    self.assertIn("b", r)

  def testIteration(self):
    r = registry.Registry("test_registry")
    r["a"] = lambda: None
    r["b"] = lambda: 4
    self.assertListEqual(list(r), ["a", "b"])

  def testLen(self):
    r = registry.Registry("test_registry")
    r["a"] = lambda: None
    self.assertLen(r, 1)
    r["b"] = lambda: 4
    self.assertLen(r, 2)
    r.deregister("b")
    self.assertLen(r, 1)

  def testTransformer(self):
    r = registry.Registry(
        "test_registry", value_transformer=lambda x, y: x + y())
    r.register(3)(lambda: 5)
    r.register(10)(lambda: 12)
    self.assertEqual(r[3], 8)
    self.assertEqual(r[10], 22)
    self.assertEqual(set(r.values()), set((8, 22)))
    self.assertEqual(set(r.items()), set(((3, 8), (10, 22))))

  def testGet(self):
    r = registry.Registry("test_registry", value_transformer=lambda k, v: v())
    r["a"] = lambda: "xyz"
    self.assertEqual(r.get("a"), "xyz")
    self.assertEqual(r.get("a", 3), "xyz")
    self.assertIsNone(r.get("b"))
    self.assertEqual(r.get("b", 3), 3)


class OpRegistryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    registry.Registries.ops._clear()

  def testOpRegistration(self):

    @registry.register_op
    class MyOp1:
      pass

    op = registry.op("MyOp1")
    self.assertIs(op, MyOp1)

  def testNamedRegistration(self):

    @registry.register_op("op2")
    class MyOp1:
      pass

    op = registry.op("op2")
    self.assertIs(op, MyOp1)

  def testUnknownop(self):
    with self.assertRaisesRegex(KeyError, "never registered"):
      registry.op("not_registered")

  def testDuplicateRegistration(self):

    @registry.register_op
    def m1():
      pass

    with self.assertRaisesRegex(KeyError, "already registered"):

      @registry.register_op("m1")
      def m2():
        pass


if __name__ == "__main__":
  absltest.main()
