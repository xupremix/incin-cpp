# Changelog

## 0.1.0 - 2026-10-29

Rename and first functional release.

### Changed

- Renamed the project from `evolcpp` / `evol_cpp` to `incin-cpp`.
- Renamed the C++ namespace from `evol` to `incin`.
- Renamed the CMake targets to `incin_cpp` (interface library, alias
  `incin::incin`) and `incin_example`.
- Moved the public header to `include/incin/tensor.hpp` with the umbrella
  `include/incin/incin.hpp`.
- Replaced the ad-hoc Makefile with CMake options, CTest, install rules, and a
  convenience Makefile.
- Replaced the stub `README.md` with build instructions, an API table, and
  compile-time examples.

### Fixed

- `Tensor` no longer inherits from `at::TensorBase` and no longer exposes a
  second, shadowing `torch::Tensor` member.
- Copy and move constructors/assignments now work; they previously referenced
  the tensor through `operator->` and the copy assignment returned a pointer.
- `reshape`, `transpose`, `permute`, `flatten`, `unsqueeze`, `squeeze`,
  `stack`, `cat`, `repeat`, `top`, `narrow`, `slice`, `select`, `select_only`,
  `chunk`, `pad`, `arange`, and `bool_not` now call libtorch instead of
  returning zero-initialized placeholders.
- `reshape` preserves the source values.
- `UnsqueezeDimShape` and `VariadicSqueezeShape` now compute correct shapes and
  reject invalid dimensions.
- `SliceShape` now keeps the rank, matching `narrow`.
- `AreAllSameAs` and the shape predicates handle empty packs.
- `pad` accepts a scalar value instead of a dtype tag.
- `matmul` accepts a `const` operand.

### Added

- Runtime value tests for every implemented operation (`tests/tensor_test.cpp`).
- Elementwise math, reductions, `item<T>()`, `allclose`, `clone`, `detach`,
  `full`/`rand`/`randn`, `from_torch` validation, and `to_torch()`.
- GitHub Actions CPU build/test workflow.
- MIT and Apache-2.0 license files, matching Incin.
