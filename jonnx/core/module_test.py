"""Test Module class."""
import dataclasses
from typing import Dict, List

from absl.testing import absltest
from jonnx.core import module


class GraphTest(absltest.TestCase):

  def test_static_field(self):

    @dataclasses.dataclass
    class A:
      a: List[str] = module.static_field(default_factory=[].copy)
      b: Dict[str, str] = module.static_field(default_factory={}.copy)

    a1 = A()
    for field_ in dataclasses.fields(a1):
      self.assertIn("static", field_.metadata)

  def test_wrap_method(self):
    func = lambda x, y: x + y
    wrapper = module.WrapMethod(func)
    f1 = wrapper
    self.assertIs(f1.method(1, 3), 4)

    func = lambda x, y: x(y)

    class A:
      wrapper = module.WrapMethod(func)

      def __call__(self, y):
        return 100 + y

    a0 = A()
    self.assertEqual(a0.wrapper(200), 300)

  def test_not_magic(self):
    self.assertEqual(module._not_magic("do_something"), True)
    self.assertEqual(module._not_magic("__init__"), False)

  def test_module_basic(self):

    class MyModule(module.Module):
      x: float
      y: float
      z: module.static_field()

      def __call__(self):
        return self.x, self.y, self.z

    m = MyModule(x=100, y=200, z=300)
    self.assertCountEqual(m(), [100, 200, 300])
    with self.assertRaisesRegex(
        dataclasses.FrozenInstanceError,
        expected_regex="cannot assign to field"):
      m.x = 400

    with self.assertRaisesRegex(
        TypeError, expected_regex="missing 1 required positional"):
      m = MyModule(x=100, y=200)

    with self.assertRaisesRegex(
        TypeError, expected_regex="got an unexpected keyword"):
      m = MyModule(x=100, y=200, z=300, d=400)


if __name__ == "__main__":
  absltest.main()
