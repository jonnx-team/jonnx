"""Test utility functions."""
from typing import Any, Sequence

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx.backend.test.case.node as backend_test_case_node
from jonnx.backend import run_model
from jax._src import test_util as jtu

random_key = jax.random.PRNGKey(0)


class ExpectationError(Exception):
  pass


def gen_random(minval=-1.0, maxval=1.0, shape=None, dtype=jnp.float32):
  output = jax.random.uniform(
      random_key, shape=shape, dtype=dtype, minval=minval, maxval=maxval)
  return output


def cosin_sim(a, b):
  a = a.astype(jnp.float32)
  b = b.astype(jnp.float32)
  a = a.flatten()
  b = b.flatten()
  cos_sim = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
  return cos_sim


def expect(self,
    node_op: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    **kwargs: Any,
):
  """From onnx/backend/test/case/node/__init__.py#L219."""
  backend_test_case_node.expect(node_op, inputs, outputs, name, **kwargs)
  # pylint: disable=protected-access
  testcase = backend_test_case_node._NodeTestCases.pop()
  # pylint: enable=protected-access
  model = testcase.model
  atol = testcase.atol
  rtol = testcase.rtol
  outputs_jax = run_model(model, inputs)
  
  self.assertCountEqual(list(outputs[0].shape), list(outputs_jax[0].shape))
  self.assertAllClose(outputs, outputs_jax, rtol=rtol, atol=atol)