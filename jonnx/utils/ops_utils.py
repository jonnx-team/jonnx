"""util functions support op def."""
from typing import Sequence, Tuple


def convert_onnx_pad_to_jax_pad(
    pads: Sequence[int]) -> Sequence[Tuple[int, int]]:
  """ONNX Pads convention is [x1_begin, x2_begin,..., x1_end, x2_end,...].

    While JAX pads convention is ([(x1_begin, x1_end), (x2_begin, x2_end),..]

    Args:
      pads: ONNX conversion pads.

    Returns:
      result: JAX convention pads.
  """
  length = len(pads)
  assert length % 2 == 0, f'length = {length}, pads = {pads}'
  result = []
  for i in range(length // 2):
    result.append((pads[i], pads[i + length // 2]))
  print('result = %s', result)
  return result
