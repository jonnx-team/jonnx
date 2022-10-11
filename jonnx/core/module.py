"""Define Base class Module of all ops.

Copy from https://github.com/patrick-kidger/equinox/blob/main/equinox/module.
"""
import abc
import dataclasses
import functools as ft
import inspect

from jax import numpy as jnp
import jax.tree_util as jtu
from jaxtyping.pytree_type import PyTree
import numpy as np


def tree_equal(*pytrees: PyTree):
  """Returns `True` if all input PyTrees are equal."""
  if len(pytrees) < 2:
    raise RuntimeError("tree_equal need at least 2 pytrees inputs.")
  flat, treedef = jtu.tree_flatten(pytrees[0])
  array_types = (jnp.ndarray, np.ndarray)
  out = True
  for pytree in pytrees[1:]:
    flat_, treedef_ = jtu.tree_flatten(pytree)
    if treedef_ != treedef:
      return False
    for elem, elem_ in zip(flat, flat_):
      if isinstance(elem, array_types):
        if isinstance(elem_, array_types):
          if (isinstance(elem, type(elem_)) or (elem.shape != elem_.shape) or
              (elem.dtype != elem_.dtype)):
            return False
          allsame = (elem == elem_).all()
          if not allsame:
            return False
          out = out & allsame
        else:
          return False
      else:
        if isinstance(elem_, array_types):
          return False
        else:
          if elem != elem_:
            return False
  return out


def static_field(**kwargs):
  """Wrap the dataclasses and add static metadata info."""
  try:
    metadata = dict(kwargs["metadata"])
  except KeyError:
    metadata = kwargs["metadata"] = {}
  if "static" in metadata:
    raise ValueError("Cannot use metadata with `static` already set.")
  metadata["static"] = True
  return dataclasses.field(**kwargs)


class WrapMethod:
  """Wrap python class as class."""

  def __init__(self, method):
    self.method = method
    if getattr(self.method, "__isabstractmethod__", False):
      self.__isabstractmethod__ = self.method.__isabstractmethod__

  def __get__(self, instance, owner):
    if instance is None:
      return self.method
    return jtu.Partial(self.method, instance)


def _not_magic(k: str) -> bool:
  return not (k.startswith("__") and k.endswith("__"))


# Inherits from abc.ABCMeta as a convenience for a common use-case.
# It's not a feature we use ourselves.
class _ModuleMeta(abc.ABCMeta):
  """decorate cls with dataclass and register it as pytree node type."""

  def __new__(cls, name, bases, dict_):
    dict_ = {
        k: WrapMethod(v) if _not_magic(k) and inspect.isfunction(v) else v
        for k, v in dict_.items()
    }
    sub_cls = super().__new__(cls, name, bases, dict_)
    has_init = sub_cls._has_dataclass_init = _has_dataclass_init(sub_cls)
    if has_init:
      init_doc = sub_cls.__init__.__doc__
    sub_cls = dataclasses.dataclass(
        eq=False, repr=False, frozen=True, init=has_init)(
            sub_cls)
    if has_init:
      sub_cls.__init__.__doc__ = init_doc
    jtu.register_pytree_node_class(sub_cls)
    return sub_cls

  def __call__(cls, *args, **kwargs):
    self = cls.__new__(cls, *args, **kwargs)

    # Defreeze it during __init__
    initable_cls = _make_initable(cls, wraps=False)
    object.__setattr__(self, "__class__", initable_cls)
    try:
      cls.__init__(self, *args, **kwargs)
    finally:
      object.__setattr__(self, "__class__", cls)

    missing_names = {
        field.name
        for field in dataclasses.fields(cls)
        if field.init and field.name not in dir(self)
    }
    if missing_names:
      raise ValueError(
          f"The following fields were not initialised during __init__: {missing_names}"
      )
    return self


@ft.lru_cache(maxsize=128)
def _make_initable(cls: _ModuleMeta, wraps: bool) -> _ModuleMeta:
  """Wrap Module as InitModule for dataclass member init."""
  if wraps:
    field_names = {
        "__module__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__annotations__",
        "__wrapped__",
    }
  else:
    field_names = {field.name for field in dataclasses.fields(cls)}

  class _InitableModule(cls):
    pass

  # Done like this to avoid dataclasses complaining about overriding setattr
  # on a frozen class.
  def __setattr__(self, name, value):  # pylint: disable=invalid-name
    if name in field_names:
      object.__setattr__(self, name, value)
    else:
      raise AttributeError(f"Cannot set attribute {name}")

  _InitableModule.__setattr__ = __setattr__

  return _InitableModule


def _has_dataclass_init(cls: _ModuleMeta) -> bool:
  if "__init__" in cls.__dict__:
    return False
  return cls._has_dataclass_init  # pylint: disable=protected-access


class Module(metaclass=_ModuleMeta):
  """Base class of operator class.

    **Fields**

    Specify all its fields at the class level (identical to
    [dataclasses](https://docs.python.org/3/library/dataclasses.html)). This
    defines
    its children as a PyTree.

    ```python
    class MyModule(Module):
        weight: jax.numpy.ndarray
        bias: jax.numpy.ndarray
        submodule: Module
    ```

    **Initialisation**

    A default `__init__` is automatically provided, which just fills in fields
    with the
    arguments passed. For example `MyModule(weight, bias, submodule)`.

    Alternatively (quite commonly) you can provide an `__init__` method
    yourself:

    ```python
    class MyModule(Module):
        weight: jax.numpy.ndarray
        bias: jax.numpy.ndarray
        submodule: Module

        def __init__(self, in_size, out_size, key):
            wkey, bkey, skey = jax.random.split(key, 3)
            self.weight = jax.random.normal(wkey, (out_size, in_size))
            self.bias = jax.random.normal(bkey, (out_size,))
            self.submodule = nn.Linear(in_size, out_size, key=skey)
    ```

    **Usage**

    After you have defined your model, then you can use it just like any other
    PyTree
    -- that just happens to have some methods attached. In particular you can
    pass it
    around across `jax.jit`, `jax.grad` etc. in exactly the way that you're used
    to.

    !!! example

        If you wanted to, then it would be completely safe to do

        ```python
        class MyModule(Module):
            ...

            @jax.jit
            def __call__(self, x):
                ...
        ```

        because `self` is just a PyTree.
  """

  _has_dataclass_init = True

  def __hash__(self):
    return hash(tuple(jtu.tree_leaves(self)))

  def tree_flatten(self):
    """Flat Module into dynamic_field and static_field."""
    dynamic_field_names = []
    dynamic_field_values = []
    static_field_names = []
    static_field_values = []
    for field_ in dataclasses.fields(self):
      name = field_.name
      try:
        value = self.__dict__[name]
      except KeyError:
        continue
      if field_.metadata.get("static", False):
        static_field_names.append(name)
        static_field_values.append(value)
      else:
        dynamic_field_names.append(name)
        dynamic_field_values.append(value)
    return tuple(dynamic_field_values), (
        tuple(dynamic_field_names),
        tuple(static_field_names),
        tuple(static_field_values),
    )

  @classmethod
  def tree_unflatten(cls, aux, dynamic_field_values):
    self = cls.__new__(cls)
    dynamic_field_names, static_field_names, static_field_values = aux
    for name, value in zip(dynamic_field_names, dynamic_field_values):
      object.__setattr__(self, name, value)
    for name, value in zip(static_field_names, static_field_values):
      object.__setattr__(self, name, value)
    return self
  
  def __eq__(self, other):
    return tree_equal(self, other)


def module_update_wrapper(wrapper: Module, wrapped) -> Module:
  """Modifies in-place, just like functools.update_wrapper."""
  cls = wrapper.__class__
  initable_cls = _make_initable(cls, wraps=True)
  object.__setattr__(wrapper, "__class__", initable_cls)
  try:
    # updated = ("__dict__",) is the default, but that's a bit much.
    # It's common/possible for wrapper and wrapped to both be classes
    # implementing __call__, in which case copying __dict__ over basically
    # just breaks the wrapper class.
    ft.update_wrapper(wrapper, wrapped, updated=())
  finally:
    object.__setattr__(wrapper, "__class__", cls)
  return wrapper
