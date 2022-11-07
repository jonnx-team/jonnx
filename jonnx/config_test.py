"""Tests for config_class."""
from absl.testing import absltest
from jonnx.config import config


class ConfigClassTest(absltest.TestCase):

  def test_basic(self):
    config.update('jonnx_runtime_profile', True)
    self.assertTrue(config.jonnx_runtime_profile)

  def test_illegal_option(self):
    with self.assertRaisesRegex(AttributeError, 'Unrecognized config option.*'):
      config.update('jonnx_illegal_option', True)


if __name__ == '__main__':
  absltest.main()
