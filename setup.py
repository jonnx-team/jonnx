"""JAX based ONNX Backend."""
import setuptools

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setuptools.setup(
    name='jonnx',
    version='0.0.1',
    description='jonnx',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={},
    scripts=[],
    install_requires=[
        'jax',
        'jaxlib',
        'onnx',
        'orbax',
        'absl-py',
        'cached_property',
        'dataclasses',
        'flax',
        'jax',
        'jaxlib',
        'numpy',
        'pyyaml',
        'tensorflow',
        'tensorstore >= 0.1.20',
        'equinox',
    ],
    extras_require={
        'test': ['pytest'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='JAX ONNX Machine Learning',
)
