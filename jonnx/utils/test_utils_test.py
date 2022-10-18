"""Tests for registry."""
from absl.testing import absltest
from jonnx.utils import test_utils


class TestUtilsTest(absltest.TestCase):

  def test_load_model_from_url(self):
    url = ('https://github.com/onnx/models/blob/'
           '81c4779096d1205edd0b809e191a924c58c38fef/'
           'mnist/model.onnx?raw=true')
    md5sum = 'bc8ad9bd19c5a058055dc18d0f089dad'
    model = test_utils.load_model_from_url(url, md5sum)

if __name__ == '__main__':
  absltest.main()
