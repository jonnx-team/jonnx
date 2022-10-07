load("//devtools/python/blaze:strict.bzl", "py_strict_test")

py_library(
    name = "backend",
    srcs = ["backend.py"],
    deps = [
        "//experimental/users/johnqiangzhang/prototype/jonnx/utils:registry",
        "//third_party/py/jax",
    ],
)

py_strict_test(
    name = "backend_test",
    srcs = ["backend_test.py"],
    deps = [
        ":backend",
        "//experimental/users/johnqiangzhang/prototype/jonnx/utils:registry",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)
