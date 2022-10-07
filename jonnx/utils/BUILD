load("//devtools/python/blaze:strict.bzl", "py_strict_test")

py_library(
    name = "registry",
    srcs = ["registry.py"],
)

py_strict_test(
    name = "registry_test",
    srcs = ["registry_test.py"],
    deps = [
        ":registry",
        "//third_party/py/absl/testing:absltest",
    ],
)
