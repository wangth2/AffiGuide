# TorchDrug extension

`torch_ext.cpp` is the TorchDrug 0.2.1 utility extension (Apache-2.0; accompanying license). The bundled `torch_ext.so` has SHA256 `6c73049ddd712710a30683fc9d20e8f5088f63af35fcdd4c5ec2e3db15a24f3c`. Its source SHA256 is `529128a2a59a495ac473b7fc39baf414d46b54ad112bb9d46dfc3494920e9c87`.

The bundled extension targets Linux x86-64, Python 3.8, PyTorch 1.12.1, CUDA 11.3, C++11 ABI 0. Original compilation used C++14, `-Ofast -fopenmp -DAT_PARALLEL_OPENMP -DCUDA_OP`, PyTorch/pybind include flags and its C++ ABI. This is a Python/PyTorch extension, unlike the device-only portable fatbins. The loader checks its exact hash and never recompiles at import. Other ABIs require a compatible extension build. The generator environment need not import this extension.
