"""orbax config."""
import itertools
import sys
from jax._src import config as jax_config

bool_env = jax_config.bool_env
int_env = jax_config.int_env


class Config(jax_config.Config):
  """Global config class for orbax library."""

  def parse_flags_with_absl(self):
    global already_configured_with_absl
    if not already_configured_with_absl:
      # Extract just the --jonnx... flags (before the first --) from argv. In some
      # environments (e.g. ipython/colab) argv might be a mess of things
      # parseable by absl and other junk.
      jonnx_argv = itertools.takewhile(lambda a: a != '--', sys.argv)
      jonnx_argv = [
          '', *(a for a in jonnx_argv if a.startswith('--jonnx_'))
      ]

      import absl.flags  # pylint: disable=g-import-not-at-top
      self.config_with_absl()
      absl.flags.FLAGS(jonnx_argv, known_only=True)
      self.complete_absl_config(absl.flags)
      already_configured_with_absl = True


config = Config()
flags = config
FLAGS = flags.FLAGS

already_configured_with_absl = False

jonnx_export_debug_validation = config.define_bool_state(
    name='jonnx_runtime_profile',
    default=bool_env('JONNX_RUNTIME_PROFILE', False),
    help=('Enable runtime profile for each onnx op.'))
