# Incin C++

[![CI](https://github.com/xupremix/incin-cpp/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/xupremix/incin-cpp/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](./LICENSE_MIT)

**compile-time shape, dtype, and device checking for tensors, on top of libtorch.**

`incin-cpp` is the C++ sibling of [Incin](https://github.com/xupremix/incin), the
Rust deep-learning framework that puts tensor invariants into the type system.
It keeps the same idea in C++17: shapes are template parameters, so the
compiler rejects a bad `matmul`, `reshape`, `squeeze`, or `slice` before the
program ever reaches libtorch.

```cpp
#include <incin/incin.hpp>

using namespace incin;

int main() {
  auto x = Tensor<Shape<2, 3>, f32, CPU<0>>::ones();
  auto w = Tensor<Shape<3, 4>, f32, CPU<0>>::ones();

  auto y = x.matmul(w);   // [2, 3] x [3, 4] -> [2, 4]. Compiles.
  y.print();

  auto bad = Tensor<Shape<5, 3>, f32, CPU<0>>::ones();
  // x.matmul(bad);       // inner dimensions 3 and 5 disagree. Does not compile.
}
```

## Why

libtorch checks shapes when a kernel launches. `incin-cpp` checks them when you
compile:

- **Static ranks and dimensions.** `Shape<2, 3>` is a type, not a runtime vector.
- **Static operation results.** `matmul` returns `Shape<2, 4>`, and
  `reshape<4, 1>()` only exists when the element count is preserved.
- **Static argument validation.** `squeeze<Dims...>` only accepts dimensions
  whose size is 1; `narrow<...>` rejects ranges past the end of a dimension.
- **No runtime overhead for the checks.** Once the program compiles, the
  operation lowers to the corresponding libtorch call.

The project is intentionally small and header-only. It is a typed C++ surface
over libtorch, not a reimplementation of the kernels.

## Build

Requirements: C++17 compiler, CMake 3.18+, and libtorch 2.5.x (CPU or CUDA).

```bash
# CPU build
curl -L -o libtorch.zip \
  'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip'
unzip -q libtorch.zip

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
./build/incin_example
```

For CUDA 12.4, use the matching archive:

```bash
curl -L -o libtorch.zip \
  'https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip'
```

CMake automatically picks up `./libtorch` when it exists. To use another
installation, pass `-DCMAKE_PREFIX_PATH=/path/to/libtorch`.

Convenience targets:

```bash
make          # configure + build
make test     # build + ctest
make run      # build + run the matmul example
make clean
```

CMake options:

| Option | Default | Meaning |
| --- | --- | --- |
| `INCIN_CPP_BUILD_TESTS` | `ON` | Build and register the test suite |
| `INCIN_CPP_BUILD_EXAMPLES` | `ON` | Build `incin_example` |

### Installing and consuming

```bash
cmake --install build --prefix /path/to/prefix
```

```cmake
find_package(incin-cpp REQUIRED)
target_link_libraries(app PRIVATE incin::incin)
```

The installed package is relocatable. It records the imported `torch` target
rather than the expanded `TORCH_LIBRARIES` list, so no build-machine paths are
baked into the export; `find_dependency(Torch)` resolves libtorch on the
consumer's machine. Add libtorch to `CMAKE_PREFIX_PATH` when configuring the
consumer:

```bash
cmake -S app -B app/build \
  -DCMAKE_PREFIX_PATH="/path/to/prefix;/path/to/libtorch"
```

See `tests/consumer/` for a complete example; CI installs the package, moves
the prefix, and builds that consumer against the relocated copy.

## API

All dimensions and devices are template parameters. The default dtype/device is
`f32` / `CPU<0>`.

```cpp
using M = Tensor<Shape<3, 4>>;          // float32 on CPU
using G = Tensor<Shape<3, 4>, f32, CUDA<0>>;
```

### Data types and devices

`i8 i16 i32 i64 u8 u16 u32 u64 f16 f32 f64 c16 c32 c64`, plus the quantized and
bit types exposed by libtorch. Devices: `CPU`, `CUDA`, `HIP`, `XLA`, `MPS`,
`Meta`, `Vulkan`, `Metal`, `XPU`, `HPU`, `VE`, `Lazy`, `IPU`, `MTIA`, and the
remaining libtorch device types.

### Operations

| Group | Operations |
| --- | --- |
| Generators | `zeros`, `ones`, `full`, `rand`, `randn`, `arange`, `zeros_like`, `ones_like`, `full_like` |
| Elementwise | `+`, `-`, `*`, `/`, scalar variants, unary `-`, `abs`, `exp`, `log`, `sqrt`, `relu`, `sigmoid`, `tanh`, `pow`, `clamp`, `softmax` |
| Reductions | `sum`, `mean`, `max`, `min`, `allclose`, `item<T>()` |
| Linear algebra | `matmul` |
| Reshaping | `reshape`, `flatten`, `unsqueeze`, `unsqueeze_dim`, `squeeze`, `transpose`, `permute`, `swap_dims` |
| Combining | `stack`, `cat` |
| Indexing | `top`, `narrow`, `slice`, `select`, `select_only`, `chunk` |
| Padding | `pad` (last two dimensions) |
| Conversion | `to_dtype`, `to_device` |
| Escape hatches | `native()`, `to_torch()`, `from_torch`, `from_torch_unchecked`, `data_ptr<T>()`, `clone`, `detach` |

`select_only<Dim, Indices...>` keeps the selected dimensions and collapses every
other dimension at index `0`; the shape reflects that. Use `select` when you
want to keep the rank of the unselected dimensions.

The shape arguments are short aliases: `Tp<K, Dim>` is a `TensorPair` used by
`repeat` and `top`, `Tr<Start, End, Dim>` is a half-open `TensorRange` used by
`narrow` and `slice`, and `Ip<Dim, Indices...>` is an `IndicesPair` used by
`select` and `select_only`.

The variadic forms `unsqueeze<Dims...>()` and `squeeze<Dims...>()` take
dimensions in increasing order, relative to the original shape. A dimension that
does not exist, or a `squeeze` dimension whose size is not `1`, is a compile
error. `top<...>()` follows the original API and narrows each named dimension
from index `0`, so it is a typed `narrow` rather than a top-k value selection.

## Compile-time examples

```cpp
using A = Tensor<Shape<2, 3>>;
using B = Tensor<Shape<3, 4>>;
using C = Tensor<Shape<2, 4>>;

A a; B b;
C c = a.matmul(b);                 // ok
// auto d = a.matmul(a);           // 3 != 2: compile error

auto flat = a.reshape<6>();        // ok: 2 * 3 == 6
// auto bad = a.reshape<5>();      // compile error: 6 != 5

Tensor<Shape<2, 1, 3>> u = a.unsqueeze<1>();
Tensor<Shape<2, 3>> back = u.squeeze<1>();   // ok: that dimension is 1
// u.squeeze<0>();                 // compile error: dimension 0 has size 2

auto r = a.narrow<Tr<0, 2, 0>>();  // rows [0, 2) -> Shape<2, 3>
// a.narrow<Tr<0, 4, 0>>();        // compile error: past the end
```

## Relationship to Incin

[Incin](https://github.com/xupremix/incin) is the Rust project. It covers typed
shapes, dtypes, devices, gradients, kernels, neural-network layers, data
loading, serialization, an LSP, and diagnostics. `incin-cpp` is a focused C++
port of the tensor-typing idea: it gives libtorch a typed shape layer and
leaves the rest of the ecosystem to libtorch for now. See [ROADMAP.md](ROADMAP.md)
for the gap between the two.

The old `evolcpp` name and `namespace evol` are gone; this is `incin-cpp` and
`namespace incin`.

## Status

Early but usable. Every operation listed above is wired to libtorch and covered
by runtime value tests, in addition to the compile-time checks. The compile-time
guarantees are themselves covered by CTest compile-fail fixtures in
`tests/compile_fail/`, which assert that invalid programs are rejected with the
expected diagnostic. There is no typed autograd, module, dataset, or
serialization layer yet.

## License

MIT OR Apache-2.0, matching Incin. See [LICENSE_MIT](LICENSE_MIT) and
[LICENSE_APACHE](LICENSE_APACHE).
