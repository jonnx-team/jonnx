"""Run mobilenet."""
import logging

from absl import app
from absl import flags
import jax
from jonnx import backend
from jonnx.utils import test_utils
import onnx

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

url = ('https://github.com/onnx/models/blob/'
       '131c99da401c757207a40189385410e238ed0934/vision/classification/'
       'mobilenet/model/mobilenetv2-7.onnx?raw=true')
md5sum = None
flags.DEFINE_string('url', url, 'onnx model url')
flags.DEFINE_string('md5sum', md5sum, 'expected md5sum to verify it.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  model = test_utils.load_model_from_url(FLAGS.url, FLAGS.md5sum)
  logging.info('download the mnist successfully.')
  model = onnx.shape_inference.infer_shapes(model)
  onnx.checker.check_model(model)
  key = jax.random.PRNGKey(0)
  inputs = jax.random.normal(key, [1, 3, 224, 224])
  results = backend.run_model(model, [inputs])
  logging.info('results is =\n %s', results)


if __name__ == '__main__':
  app.run(main)
