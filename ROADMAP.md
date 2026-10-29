# Roadmap

`incin-cpp` currently implements the typed tensor surface. The next steps move
it closer to what [Incin](https://github.com/xupremix/incin) provides.

## Near term

- [ ] Add a CUDA job to CI (the current workflow is CPU-only).
- [x] Add compile-fail tests: a small set of intentionally invalid programs that
      must not compile, checked by CMake/CTest.
- [ ] Add broadcasting support (`broadcast_to`, `expand`) with compile-time
      compatibility checks.
- [ ] Add reductions over a chosen dimension, not just full-tensor reductions.
- [ ] Add `argmax` / `argmin` / `sort` / `flip` / `any` / `all` with typed
      result shapes.
- [ ] Add `gather`, `scatter`, `scatter_add`, `masked_fill`, and `where`.
- [ ] Add `load` / `save` for `torch::save` and `torch::load`, plus safetensors.

## Medium term

- [ ] Typed autograd: a `Grad` tag on `Tensor` and a typed `backward()` result.
- [ ] Neural-network modules mirroring Incin: `Linear`, `Conv2d`, activations,
      normalization, losses, and an optimizer interface.
- [ ] Dataset and data-loader abstractions over libtorch.
- [x] A relocatable CMake package config that downstream projects can
      `find_package` (consumer smoke-tested in CI).
- [ ] Documentation site generated from the headers.

## Long term

- [ ] A separately compiled backend boundary so consumers do not need the full
      libtorch headers for every translation unit.
- [ ] Compile-time device compatibility for kernel arguments.
- [ ] Optional CUDA/HIP kernel implementations that do not route through
      libtorch.
- [ ] Benchmarks against handwritten libtorch code to confirm that the typed
      layer is zero-cost.

## Not planned

- Reimplementing libtorch's kernel catalog.
- Runtime dynamic shapes as the default; static shapes stay the point.
- A Python binding. Use libtorch's own Python surface for that.
