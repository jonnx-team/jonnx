import os

from absl import app
from absl import flags
from absl import logging
import jax
from jax import numpy as jnp
import jonnx
import onnx
from jonnx.utils import test_utils
from jonnx import backend


url = ('https://github.com/onnx/models/blob/'
                    '81c4779096d1205edd0b809e191a924c58c38fef/'
                    'mnist/model.onnx?raw=true')
md5sum = 'bc8ad9bd19c5a058055dc18d0f089dad'
flags.DEFINE_string('url', url, 'onnx model url')
flags.DEFINE_string('md5sum', md5sum, "expected md5sum to verify it.")


FLAGS = flags.FLAGS

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  url = FLAGS.url
  md5sum = FLAGS.md5sum
  model = test_utils.load_model_from_url(url, md5sum)
  logging.info(f"download the mnist successfully.")
  model = onnx.shape_inference.infer_shapes(model)
  onnx.checker.check_model(model)
  key = jax.random.PRNGKey(0)
  inputs = jax.random.normal(key, [1, 1, 28, 28])
  results = backend.run_model(model, [inputs])
  logging.info("results is =\n %s", results)

if __name__ == "__main__":
  app.run(main)