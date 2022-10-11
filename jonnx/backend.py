"""Create the JAX based ONNX Backend."""
from typing import Any

import jax
from jonnx.core import Graph
from jonnx.utils import registry
import onnx
from onnx import helper
from onnx import ModelProto
from onnx import numpy_helper
from onnx.backend.base import Backend
from onnx.backend.base import BackendRep


def _asarray(proto):
  return numpy_helper.to_array(proto).reshape(tuple(proto.dims))


attr_types = dict(onnx.AttributeProto.AttributeType.items())  # type: ignore
attribute_handlers = {
    attr_types['FLOAT']: lambda a: a.f,
    attr_types['INT']: lambda a: a.i,
    attr_types['STRING']: lambda a: a.s,
    attr_types['TENSOR']: lambda a: _asarray(a.t),
    attr_types['FLOATS']: lambda a: a.floats,
    attr_types['INTS']: lambda a: a.ints,
    attr_types['STRINGS']: lambda a: a.strings,
    attr_types['TENSORS']: lambda a: [_asarray(x) for x in a.tensors],
}


class JaxBackendRep(BackendRep):
  """the handle that preparing to execut a model repeatedly.

  Users will then pass inputs to the run function of
  BackendRep to retrieve the corresponding results.
  """

  def __init__(self, model=None):
    super(JaxBackendRep, self).__init__()
    g = Graph(model.graph)
    new_graph_proto = g.export()
    self.model = helper.make_model(new_graph_proto)

  def run(self, inputs, device='CPU', mode='jit', **kwargs):
    """run the model."""

    # TODO(johnqiangzhang):  Add the reference count to release unused tensors.
    def jax_func(model, inputs):
      vals = dict({n.name: a for n, a in zip(model.graph.input, inputs)},
                  **{n.name: _asarray(n) for n in model.graph.initializer})
      for node in model.graph.node:
        args = (vals[name] for name in node.input)
        attrs = {a.name: attribute_handlers[a.type](a) for a in node.attribute}
        outputs = registry.op(node.op_type)(*args, **attrs)
        for name, output in zip(node.output, outputs):
          vals[name] = output
      return [vals[n.name] for n in model.graph.output]

    mode = kwargs.get('mode', 'jit')
    print('running mode: ', mode)
    predict = lambda inputs: jax_func(self.model, inputs)
    if mode == 'jit':
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
    assert backend is not None
    return backend.run(inputs)

  @classmethod
  def supports_device(cls, device: str) -> bool:
    """check which particular device support."""
    return device in ('CPU', 'CUDA', 'TPU')
