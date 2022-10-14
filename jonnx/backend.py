"""Create the JAX based ONNX Backend."""
from typing import Any

import jax
from jonnx.core import graph
import onnx
from onnx import ModelProto
from onnx import numpy_helper
from onnx.backend.base import Backend
from onnx.backend.base import BackendRep


def _asarray(proto):
  return numpy_helper.to_array(proto).reshape(tuple(proto.dims))


class JaxBackendRep(BackendRep):
  """the handle that preparing to execut a model repeatedly.

  Users will then pass inputs to the run function of
  BackendRep to retrieve the corresponding results.
  """

  def __init__(self, model=None):
    super(JaxBackendRep, self).__init__()
    self.graph = graph.Graph(model.graph)

  def run(self, inputs, device='CPU', mode='jit', **kwargs):
    """run the model."""
    node_list = self.graph.topological_sort()
    ref_dict = self.graph.create_ref_dict()

    # TODO(johnqiangzhang):  Add the reference count to release unused tensors.
    def jax_func(inputs):

      vals = dict({name: a for name, a in zip(self.graph.input, inputs)},
                  **self.graph.initializer_dict)

      for nd_name in node_list:
        node_op = self.graph.node_dict[nd_name]
        args = (vals[name] for name in node_op.input)
        outputs = node_op(*args)
        for name, output in zip(node_op.output, outputs):
          vals[name] = output
        for name in node_op.input:
          if name in ref_dict:
            if ref_dict[name] > 1:
              ref_dict[name] -= 1
            else:
              del ref_dict[name]
              del vals[name]
      return [vals[name] for name in self.graph.output]

    mode = kwargs.get('mode', 'jit')
    # pylint: disable=unnecessary-lambda
    predict = lambda inputs: jax_func(inputs)

    if mode == 'return_jax_func_only':
      return predict
    elif mode == 'jit':
      predict = jax.jit(predict)
      return predict(inputs)


class JaxBackend(Backend):
  """Jax Backend demo for ONNX."""

  @classmethod
  def prepare(cls, model, device, **kwargs):
    """Create the BackendRep obj."""
    onnx.checker.check_model(model)
    backend_rep = JaxBackendRep(model)
    return backend_rep

  @classmethod
  def run_model(cls,
                model: ModelProto,
                inputs: Any,
                device: str = 'CPU',
                **kwargs: Any):
    backend = cls.prepare(model, device, **kwargs)
    return backend.run(inputs, **kwargs)

  @classmethod
  def supports_device(cls, device: str) -> bool:
    """check which particular device support."""
    return device in ('CPU', 'CUDA', 'TPU')


run_model = JaxBackend.run_model
