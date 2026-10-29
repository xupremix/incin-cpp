#pragma once
#ifndef INCIN_TENSOR_HPP
#define INCIN_TENSOR_HPP

// incin-cpp: compile-time shape, dtype, and device checking for libtorch.
//
// Every operation is expressed in terms of types. A matmul whose inner
// dimensions disagree, a reshape that changes the number of elements, or a
// slice that runs past the end of a dimension is rejected by the compiler.
// Whatever survives that check is lowered to a single libtorch call.

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace incin {

///
///
/// DEVICE
///
///

namespace device {

#define INCIN_DEF_DEVICE(name, value)                  \
  template <int64_t N = 0>                             \
  struct name {                                        \
    static constexpr torch::DeviceType DEVICE = value; \
    static constexpr int64_t INDEX = N;                \
  };

INCIN_DEF_DEVICE(CPU, torch::kCPU);
INCIN_DEF_DEVICE(CUDA, torch::kCUDA);
INCIN_DEF_DEVICE(HIP, torch::kHIP);
INCIN_DEF_DEVICE(FPGA, torch::kFPGA);
INCIN_DEF_DEVICE(MAIA, torch::kMAIA);
INCIN_DEF_DEVICE(XLA, torch::kXLA);
INCIN_DEF_DEVICE(MPS, torch::kMPS);
INCIN_DEF_DEVICE(Meta, torch::kMeta);
INCIN_DEF_DEVICE(Vulkan, torch::kVulkan);
INCIN_DEF_DEVICE(Metal, torch::kMetal);
INCIN_DEF_DEVICE(XPU, torch::kXPU);
INCIN_DEF_DEVICE(HPU, torch::kHPU);
INCIN_DEF_DEVICE(VE, torch::kVE);
INCIN_DEF_DEVICE(Lazy, torch::kLazy);
INCIN_DEF_DEVICE(IPU, torch::kIPU);
INCIN_DEF_DEVICE(MTIA, torch::kMTIA);
INCIN_DEF_DEVICE(PrivateUse1, torch::kPrivateUse1);
INCIN_DEF_DEVICE(OPENGL, torch::DeviceType::OPENGL);
INCIN_DEF_DEVICE(OPENCL, torch::DeviceType::OPENCL);
INCIN_DEF_DEVICE(IDEEP, torch::DeviceType::IDEEP);
INCIN_DEF_DEVICE(MKLDNN, torch::DeviceType::MKLDNN);

#undef INCIN_DEF_DEVICE

}  // namespace device

template <int64_t N = 0>
using CPU = device::CPU<N>;
template <int64_t N = 0>
using CUDA = device::CUDA<N>;
template <int64_t N = 0>
using HIP = device::HIP<N>;
template <int64_t N = 0>
using FPGA = device::FPGA<N>;
template <int64_t N = 0>
using MAIA = device::MAIA<N>;
template <int64_t N = 0>
using XLA = device::XLA<N>;
template <int64_t N = 0>
using MPS = device::MPS<N>;
template <int64_t N = 0>
using Meta = device::Meta<N>;
template <int64_t N = 0>
using Vulkan = device::Vulkan<N>;
template <int64_t N = 0>
using Metal = device::Metal<N>;
template <int64_t N = 0>
using XPU = device::XPU<N>;
template <int64_t N = 0>
using HPU = device::HPU<N>;
template <int64_t N = 0>
using VE = device::VE<N>;
template <int64_t N = 0>
using Lazy = device::Lazy<N>;
template <int64_t N = 0>
using IPU = device::IPU<N>;
template <int64_t N = 0>
using MTIA = device::MTIA<N>;
template <int64_t N = 0>
using PrivateUse1 = device::PrivateUse1<N>;
template <int64_t N = 0>
using OPENGL = device::OPENGL<N>;
template <int64_t N = 0>
using OPENCL = device::OPENCL<N>;
template <int64_t N = 0>
using IDEEP = device::IDEEP<N>;
template <int64_t N = 0>
using MKLDNN = device::MKLDNN<N>;

///
///
/// DTYPE
///
///

namespace dtype {

#define INCIN_DEF_DTYPE(name, value, size)              \
  struct alignas(size) name {                           \
    static constexpr torch::Dtype DTYPE = torch::value; \
    static constexpr size_t SIZE = size;                \
  };

INCIN_DEF_DTYPE(i8, kInt8, 1);
INCIN_DEF_DTYPE(i16, kInt16, 2);
INCIN_DEF_DTYPE(i32, kInt32, 4);
INCIN_DEF_DTYPE(i64, kInt64, 8);

INCIN_DEF_DTYPE(u8, kUInt8, 1);
INCIN_DEF_DTYPE(u16, kUInt16, 2);
INCIN_DEF_DTYPE(u32, kUInt32, 4);
INCIN_DEF_DTYPE(u64, kUInt64, 8);

INCIN_DEF_DTYPE(f16, kFloat16, 2);
INCIN_DEF_DTYPE(f32, kFloat32, 4);
INCIN_DEF_DTYPE(f64, kFloat64, 8);
INCIN_DEF_DTYPE(f128, kLong, 16);

INCIN_DEF_DTYPE(byte, kByte, 1);

INCIN_DEF_DTYPE(c16, kComplexHalf, 4);
INCIN_DEF_DTYPE(c32, kComplexFloat, 8);
INCIN_DEF_DTYPE(c64, kComplexDouble, 16);

INCIN_DEF_DTYPE(q8, kQInt8, 1);
INCIN_DEF_DTYPE(q32, kQInt32, 2);
INCIN_DEF_DTYPE(q2x4, kQUInt2x4, 1);
INCIN_DEF_DTYPE(q4x2, kQUInt4x2, 1);

INCIN_DEF_DTYPE(bits8, kBits8, 1);
INCIN_DEF_DTYPE(bits16, kBits16, 2);
INCIN_DEF_DTYPE(bits2x4, kBits2x4, 1);
INCIN_DEF_DTYPE(bits4x2, kBits4x2, 1);
INCIN_DEF_DTYPE(bits1x8, kBits1x8, 1);

#undef INCIN_DEF_DTYPE

template <typename T>
struct alignas(sizeof(T)) DType;

#define INCIN_DEF_DTYPE_IMPL(name)                     \
  template <>                                          \
  struct alignas(name::SIZE) DType<name> {             \
    static constexpr torch::Dtype DTYPE = name::DTYPE; \
    static constexpr size_t SIZE = name::SIZE;         \
  };

#define INCIN_DEF_DTYPE_IMPL_BASE(name, val)          \
  template <>                                         \
  struct alignas(sizeof(name)) DType<name> {          \
    static constexpr torch::Dtype DTYPE = torch::val; \
    static constexpr size_t SIZE = sizeof(name);      \
  };

INCIN_DEF_DTYPE_IMPL(i8);
INCIN_DEF_DTYPE_IMPL(i16);
INCIN_DEF_DTYPE_IMPL(i32);
INCIN_DEF_DTYPE_IMPL(i64);

INCIN_DEF_DTYPE_IMPL(u8);
INCIN_DEF_DTYPE_IMPL(u16);
INCIN_DEF_DTYPE_IMPL(u32);
INCIN_DEF_DTYPE_IMPL(u64);

INCIN_DEF_DTYPE_IMPL(f16);
INCIN_DEF_DTYPE_IMPL(f32);
INCIN_DEF_DTYPE_IMPL(f64);
INCIN_DEF_DTYPE_IMPL(f128);

INCIN_DEF_DTYPE_IMPL(byte);

INCIN_DEF_DTYPE_IMPL(c16);
INCIN_DEF_DTYPE_IMPL(c32);
INCIN_DEF_DTYPE_IMPL(c64);

INCIN_DEF_DTYPE_IMPL(q8);
INCIN_DEF_DTYPE_IMPL(q32);
INCIN_DEF_DTYPE_IMPL(q2x4);
INCIN_DEF_DTYPE_IMPL(q4x2);

INCIN_DEF_DTYPE_IMPL(bits8);
INCIN_DEF_DTYPE_IMPL(bits16);
INCIN_DEF_DTYPE_IMPL(bits2x4);
INCIN_DEF_DTYPE_IMPL(bits4x2);
INCIN_DEF_DTYPE_IMPL(bits1x8);

#undef INCIN_DEF_DTYPE_IMPL

INCIN_DEF_DTYPE_IMPL_BASE(bool, kBool);
INCIN_DEF_DTYPE_IMPL_BASE(char, kChar);

INCIN_DEF_DTYPE_IMPL_BASE(int, kInt);
INCIN_DEF_DTYPE_IMPL_BASE(int8_t, kInt8);
INCIN_DEF_DTYPE_IMPL_BASE(int16_t, kInt16);
INCIN_DEF_DTYPE_IMPL_BASE(int64_t, kInt64);

INCIN_DEF_DTYPE_IMPL_BASE(unsigned int, kUInt32);
INCIN_DEF_DTYPE_IMPL_BASE(uint8_t, kUInt8);
INCIN_DEF_DTYPE_IMPL_BASE(uint16_t, kUInt16);
INCIN_DEF_DTYPE_IMPL_BASE(uint64_t, kUInt64);

INCIN_DEF_DTYPE_IMPL_BASE(float, kFloat);
INCIN_DEF_DTYPE_IMPL_BASE(double, kDouble);
INCIN_DEF_DTYPE_IMPL_BASE(long double, kLong);
INCIN_DEF_DTYPE_IMPL_BASE(long long, kLong);
INCIN_DEF_DTYPE_IMPL_BASE(unsigned long long, kLong);

#undef INCIN_DEF_DTYPE_IMPL_BASE

}  // namespace dtype

using i8 = dtype::i8;
using i16 = dtype::i16;
using i32 = dtype::i32;
using i64 = dtype::i64;

using u8 = dtype::u8;
using u16 = dtype::u16;
using u32 = dtype::u32;
using u64 = dtype::u64;

using f16 = dtype::f16;
using f32 = dtype::f32;
using f64 = dtype::f64;
using f128 = dtype::f128;

using byte = dtype::byte;

using c16 = dtype::c16;
using c32 = dtype::c32;
using c64 = dtype::c64;

using q8 = dtype::q8;
using q32 = dtype::q32;
using q2x4 = dtype::q2x4;
using q4x2 = dtype::q4x2;

using bits8 = dtype::bits8;
using bits16 = dtype::bits16;
using bits2x4 = dtype::bits2x4;
using bits4x2 = dtype::bits4x2;
using bits1x8 = dtype::bits1x8;

///
///
/// SHAPE
///
///

template <int64_t... Dims>
struct Shape;

template <>
struct Shape<> {
  static constexpr int64_t DIMS = 0;
  static constexpr int64_t NELEMS = 1;
  static constexpr std::array<int64_t, DIMS> SHAPE_DIMS = {};
  template <int64_t... NewDims>
  using prepend = Shape<NewDims...>;
  template <int64_t... NewDims>
  using append = Shape<NewDims...>;
};

template <int64_t First>
struct Shape<First> {
  static constexpr int64_t DIMS = 1;
  static constexpr int64_t NELEMS = First;
  static constexpr int64_t LAST = First;
  static constexpr int64_t PENULTIMATE = First;
  static constexpr std::array<int64_t, DIMS> SHAPE_DIMS = {First};
  template <int64_t... NewDims>
  using prepend = Shape<NewDims..., First>;
  template <int64_t... NewDims>
  using append = Shape<First, NewDims...>;
};

template <int64_t First, int64_t Second>
struct Shape<First, Second> {
  static constexpr int64_t DIMS = 2;
  static constexpr int64_t NELEMS = First * Second;
  static constexpr int64_t LAST = Second;
  static constexpr int64_t PENULTIMATE = First;
  static constexpr std::array<int64_t, DIMS> SHAPE_DIMS = {First, Second};
  using AllButLastTwo = Shape<>;
  template <int64_t... NewDims>
  using prepend = Shape<NewDims..., First, Second>;
  template <int64_t... NewDims>
  using append = Shape<First, Second, NewDims...>;
};

template <int64_t First, int64_t Second, int64_t... Rest>
struct Shape<First, Second, Rest...> {
  static constexpr int64_t DIMS = 2 + sizeof...(Rest);
  static constexpr int64_t NELEMS = First * Second * Shape<Rest...>::NELEMS;
  static constexpr int64_t LAST = Shape<Rest...>::LAST;
  static constexpr int64_t PENULTIMATE = Shape<Second, Rest...>::PENULTIMATE;
  static constexpr std::array<int64_t, DIMS> SHAPE_DIMS = {First, Second,
                                                           Rest...};
  using AllButLastTwo =
      typename Shape<Second, Rest...>::AllButLastTwo::template prepend<First>;
  template <int64_t... NewDims>
  using prepend = Shape<NewDims..., First, Second, Rest...>;
  template <int64_t... NewDims>
  using append = Shape<First, Second, Rest..., NewDims...>;
};

///
/// TensorPair<Dim, K> / Tp
///
template <int64_t DimT, int64_t KT>
struct TensorPair {
  static constexpr int64_t DIM = DimT;
  static constexpr int64_t K = KT;
};
template <int64_t Dim, int64_t K>
using Tp = TensorPair<Dim, K>;

///
/// TensorRange<Start, End, Dim> / Tr
///
template <int64_t StartT, int64_t EndT, int64_t DimT>
struct TensorRange {
  static constexpr int64_t START = StartT;
  static constexpr int64_t END = EndT;
  static constexpr int64_t DIM = DimT;
};
template <int64_t Start, int64_t End, int64_t Dim>
using Tr = TensorRange<Start, End, Dim>;

///
/// IndicesPair<Dim, Indices...> / Ip
///
template <int64_t Dim, int64_t... I>
struct IndicesPair {
  static constexpr int64_t DIM = Dim;
  using Indices = Shape<I...>;
};
template <int64_t Dim, int64_t... I>
using Ip = IndicesPair<Dim, I...>;

///
///
/// UTILS
///
///

template <typename P1, typename P2>
struct IsSameAsPack : std::false_type {};

template <>
struct IsSameAsPack<Shape<>, Shape<>> : std::true_type {};

template <int64_t First1, int64_t... Rest1>
struct IsSameAsPack<Shape<First1, Rest1...>, Shape<>> : std::false_type {};

template <int64_t First2, int64_t... Rest2>
struct IsSameAsPack<Shape<>, Shape<First2, Rest2...>> : std::false_type {};

template <int64_t First1, int64_t First2>
struct IsSameAsPack<Shape<First1>, Shape<First2>>
    : std::bool_constant<First1 == First2> {};

template <int64_t First1, int64_t... Rest1, int64_t First2, int64_t... Rest2>
struct IsSameAsPack<Shape<First1, Rest1...>, Shape<First2, Rest2...>>
    : std::bool_constant<
          (First1 == First2) &&
          IsSameAsPack<Shape<Rest1...>, Shape<Rest2...>>::value> {};

template <int64_t Idx, typename P1, typename P2>
struct IsCatCompatible;

template <int64_t First1, int64_t First2>
struct IsCatCompatible<0, Shape<First1>, Shape<First2>> : std::true_type {};

template <int64_t First1, int64_t... Rest1, int64_t First2, int64_t... Rest2>
struct IsCatCompatible<0, Shape<First1, Rest1...>, Shape<First2, Rest2...>>
    : std::bool_constant<
          IsSameAsPack<Shape<Rest1...>, Shape<Rest2...>>::value> {};

template <int64_t Idx, int64_t First1, int64_t... Rest1, int64_t First2,
          int64_t... Rest2>
struct IsCatCompatible<Idx, Shape<First1, Rest1...>, Shape<First2, Rest2...>>
    : std::bool_constant<
          (First1 == First2) &&
          IsCatCompatible<Idx - 1, Shape<Rest1...>, Shape<Rest2...>>::value> {};

template <int64_t Idx, int64_t Dim, typename T>
struct SetDimAtIdx;

template <int64_t Dim, int64_t First, int64_t... Dims>
struct SetDimAtIdx<0, Dim, Shape<First, Dims...>> {
  using type = Shape<Dim, Dims...>;
};

template <int64_t Idx, int64_t Dim, int64_t First, int64_t... Rest>
struct SetDimAtIdx<Idx, Dim, Shape<First, Rest...>> {
  static_assert(Idx <= sizeof...(Rest), "Index out of bounds.");
  using type =
      typename SetDimAtIdx<Idx - 1, Dim,
                           Shape<Rest...>>::type::template prepend<First>;
};

template <int64_t N, typename T>
struct IsNotInPack : std::true_type {};

template <int64_t N, int64_t First>
struct IsNotInPack<N, Shape<First>> : std::bool_constant<N != First> {};

template <int64_t N, int64_t First, int64_t... Rest>
struct IsNotInPack<N, Shape<First, Rest...>>
    : std::bool_constant<N != First && IsNotInPack<N, Shape<Rest...>>::value> {
};

template <typename T>
struct AreAllUnique;

template <>
struct AreAllUnique<Shape<>> : std::true_type {};

template <int64_t First>
struct AreAllUnique<Shape<First>> : std::true_type {};

template <int64_t First, int64_t... Rest>
struct AreAllUnique<Shape<First, Rest...>>
    : std::bool_constant<IsNotInPack<First, Shape<Rest...>>::value &&
                         AreAllUnique<Shape<Rest...>>::value> {};

template <int64_t N, typename T>
struct AreAllLessThanN;

template <int64_t N>
struct AreAllLessThanN<N, Shape<>> : std::true_type {};

template <int64_t N, int64_t First>
struct AreAllLessThanN<N, Shape<First>> : std::bool_constant<(First < N)> {};

template <int64_t N, int64_t First, int64_t... Rest>
struct AreAllLessThanN<N, Shape<First, Rest...>>
    : std::bool_constant<(First < N) &&
                         AreAllLessThanN<N, Shape<Rest...>>::value> {};

template <typename As, typename... T>
struct AreAllSameAs;

template <typename As>
struct AreAllSameAs<As> : std::true_type {};

template <typename As, typename First>
struct AreAllSameAs<As, First> : std::bool_constant<std::is_same_v<As, First>> {
};

template <typename As, typename First, typename... Rest>
struct AreAllSameAs<As, First, Rest...>
    : std::bool_constant<std::is_same_v<As, First> &&
                         AreAllSameAs<As, Rest...>::value> {};

template <int64_t... Dims>
struct AreAllInIncreasingOrder;

template <>
struct AreAllInIncreasingOrder<> : std::true_type {};

template <int64_t First>
struct AreAllInIncreasingOrder<First> : std::true_type {};

template <int64_t First, int64_t Second>
struct AreAllInIncreasingOrder<First, Second>
    : std::bool_constant<(First < Second)> {};

template <int64_t First, int64_t Second, int64_t... Rest>
struct AreAllInIncreasingOrder<First, Second, Rest...>
    : std::bool_constant<(First < Second) &&
                         AreAllInIncreasingOrder<Second, Rest...>::value> {};

template <typename... TensorRanges>
struct AreTensorRangesAllInDecreasingOrder;

template <>
struct AreTensorRangesAllInDecreasingOrder<> : std::true_type {};

template <typename First>
struct AreTensorRangesAllInDecreasingOrder<First> : std::true_type {};

template <typename First, typename Second>
struct AreTensorRangesAllInDecreasingOrder<First, Second>
    : std::bool_constant<(Second::DIM < First::DIM)> {};

template <typename First, typename Second, typename... Rest>
struct AreTensorRangesAllInDecreasingOrder<First, Second, Rest...>
    : std::bool_constant<
          (Second::DIM < First::DIM) &&
          AreTensorRangesAllInDecreasingOrder<Second, Rest...>::value> {};

template <typename... TensorRanges>
struct AreTensorRangesAllInIncreasingOrder;

template <>
struct AreTensorRangesAllInIncreasingOrder<> : std::true_type {};

template <typename First>
struct AreTensorRangesAllInIncreasingOrder<First> : std::true_type {};

template <typename First, typename Second>
struct AreTensorRangesAllInIncreasingOrder<First, Second>
    : std::bool_constant<(First::DIM < Second::DIM)> {};

template <typename First, typename Second, typename... Rest>
struct AreTensorRangesAllInIncreasingOrder<First, Second, Rest...>
    : std::bool_constant<
          (First::DIM < Second::DIM) &&
          AreTensorRangesAllInIncreasingOrder<Second, Rest...>::value> {};

template <typename... RepeatPairs>
struct AreTensorPairDimsUnique;

template <>
struct AreTensorPairDimsUnique<> : std::true_type {};

template <typename First>
struct AreTensorPairDimsUnique<First> : std::true_type {};

template <typename First, typename Second>
struct AreTensorPairDimsUnique<First, Second>
    : std::bool_constant<(First::DIM != Second::DIM)> {};

template <typename First, typename Second, typename... Rest>
struct AreTensorPairDimsUnique<First, Second, Rest...>
    : std::bool_constant<(First::DIM != Second::DIM) &&
                         AreTensorPairDimsUnique<Second, Rest...>::value> {};

template <typename S>
struct ShapeToVector;

template <int64_t... Dims>
struct ShapeToVector<Shape<Dims...>> {
  static std::vector<int64_t> get() { return {Dims...}; }
};

///
///
/// SHAPE GENERATORS
///
///

template <typename S, int64_t Idx, int64_t Mul>
struct CatArrayShape;

template <int64_t Mul>
struct CatArrayShape<Shape<>, 0, Mul> {
  using type = Shape<>;
};

template <int64_t First, int64_t Mul>
struct CatArrayShape<Shape<First>, 0, Mul> {
  using type = Shape<First * Mul>;
};

template <int64_t First, int64_t... Rest, int64_t Mul>
struct CatArrayShape<Shape<First, Rest...>, 0, Mul> {
  using type = Shape<First * Mul, Rest...>;
};

template <int64_t First, int64_t... Rest, int64_t Idx, int64_t Mul>
struct CatArrayShape<Shape<First, Rest...>, Idx, Mul> {
  static_assert(Idx <= static_cast<int64_t>(sizeof...(Rest)),
                "Index out of bounds.");
  using type = typename CatArrayShape<Shape<Rest...>, Idx - 1,
                                      Mul>::type::template prepend<First>;
};

template <int64_t Idx, typename ShapeType, typename... Shapes>
struct CatShape;

template <int64_t Idx, typename ShapeType>
struct CatShape<Idx, ShapeType> {
  static_assert(Idx < ShapeType::DIMS, "Index out of bounds.");
  using type = ShapeType;
};

template <int64_t Idx, typename ShapeType, typename FirstShape>
struct CatShape<Idx, ShapeType, FirstShape> {
  static_assert(Idx < ShapeType::DIMS, "Index out of bounds.");
  static_assert(FirstShape::DIMS == ShapeType::DIMS,
                "Shapes must have the same number of dimensions.");
  static_assert(
      IsCatCompatible<Idx, ShapeType, FirstShape>::value,
      "Shapes must have the same dimensions except for the one at Idx.");
  static constexpr int64_t new_dim =
      ShapeType::SHAPE_DIMS[Idx] + FirstShape::SHAPE_DIMS[Idx];
  using type = typename SetDimAtIdx<Idx, new_dim, ShapeType>::type;
};

template <int64_t Idx, typename ShapeType, typename FirstShape,
          typename... RestShapes>
struct CatShape<Idx, ShapeType, FirstShape, RestShapes...> {
  static_assert(Idx < ShapeType::DIMS, "Index out of bounds.");
  static_assert(FirstShape::DIMS == ShapeType::DIMS,
                "Shapes must have the same number of dimensions.");
  static_assert(
      IsCatCompatible<Idx, ShapeType, FirstShape>::value,
      "Shapes must have the same dimensions except for the one at Idx.");
  static constexpr int64_t new_dim =
      FirstShape::SHAPE_DIMS[Idx] +
      CatShape<Idx, ShapeType, RestShapes...>::new_dim;
  using type = typename SetDimAtIdx<Idx, new_dim, ShapeType>::type;
};

template <typename S1, typename S2>
struct MatmulShape {
  static_assert(S1::DIMS == S2::DIMS,
                "Shape mismatch: both operands must have the same rank.");
  static_assert(S1::DIMS > 1,
                "Shape mismatch: matmul requires at least 2 dimensions.");
  static_assert(S2::DIMS > 1,
                "Shape mismatch: matmul requires at least 2 dimensions.");
  static_assert(
      std::is_same_v<typename S1::AllButLastTwo, typename S2::AllButLastTwo>,
      "Shape mismatch: all dimensions except the last two must be equal.");
  static_assert(S1::LAST == S2::PENULTIMATE,
                "Shape mismatch: the last dimension of the left operand must "
                "match the penultimate dimension of the right operand.");
  using type = typename S1::AllButLastTwo::template append<
      S1::PENULTIMATE>::template append<S2::LAST>;
};

template <typename S1, typename S2>
struct TransposeShape;

template <int64_t First, typename S2>
struct TransposeShape<Shape<First>, S2> {
  static_assert(First >= 0 && First < S2::DIMS,
                "The permutation index is out of bounds.");
  using type = Shape<S2::SHAPE_DIMS[First]>;
};

template <int64_t First, int64_t... Rest, typename S2>
struct TransposeShape<Shape<First, Rest...>, S2> {
  static_assert(First >= 0 && First < S2::DIMS,
                "The permutation index is out of bounds.");
  using type = typename TransposeShape<
      Shape<Rest...>, S2>::type::template prepend<S2::SHAPE_DIMS[First]>;
};

// Insert `Value` at index `Dim`, shifting the remaining dimensions right.
// The extra bool disambiguates the Dim == 0 case from the recursive case.
template <int64_t Dim, int64_t Value, typename S, bool = (Dim > 0)>
struct InsertDimShape;

template <int64_t Value, typename S>
struct InsertDimShape<0, Value, S, false> {
  using type = typename S::template prepend<Value>;
};

template <int64_t Dim, int64_t Value, int64_t First, int64_t... Rest>
struct InsertDimShape<Dim, Value, Shape<First, Rest...>, true> {
  static_assert(Dim > 0 && Dim <= 1 + static_cast<int64_t>(sizeof...(Rest)),
                "Dimension out of bounds.");
  using type =
      typename InsertDimShape<Dim - 1, Value,
                              Shape<Rest...>>::type::template prepend<First>;
};

// Insert a single dimension of size 1 at index Dim.
template <int64_t Dim, typename S>
struct UnsqueezeShape {
  using type = typename InsertDimShape<Dim, 1, S>::type;
};

template <int64_t Dim, typename S>
struct UnsqueezeDimShape {
  using type = typename InsertDimShape<Dim, 1, S>::type;
};

template <int64_t Curr, typename D, typename S>
struct VariadicUnsqueezeShape;

template <int64_t Curr, typename S>
struct VariadicUnsqueezeShape<Curr, Shape<>, S> {
  using type = S;
};

template <int64_t Curr, int64_t First, int64_t... Rest, typename S>
struct VariadicUnsqueezeShape<Curr, Shape<First, Rest...>, S> {
  static_assert(AreAllInIncreasingOrder<First, Rest...>::value,
                "The provided dimensions must be in increasing order.");
  using type = typename VariadicUnsqueezeShape<
      Curr + 1, Shape<Rest...>,
      typename UnsqueezeShape<First + Curr, S>::type>::type;
};

// Remove the dimension at index Idx.
template <int64_t Idx, typename S, bool = (Idx > 0)>
struct DropDimAt;

template <int64_t First, int64_t... Rest>
struct DropDimAt<0, Shape<First, Rest...>, false> {
  using type = Shape<Rest...>;
};

template <int64_t Idx, int64_t First, int64_t... Rest>
struct DropDimAt<Idx, Shape<First, Rest...>, true> {
  static_assert(Idx > 0 && Idx < 1 + static_cast<int64_t>(sizeof...(Rest)),
                "Dimension out of bounds.");
  using type =
      typename DropDimAt<Idx - 1,
                         Shape<Rest...>>::type::template prepend<First>;
};

template <typename T>
struct SqueezeShape;

template <>
struct SqueezeShape<Shape<>> {
  using type = Shape<>;
};

template <int64_t First>
struct SqueezeShape<Shape<First>> {
  using type =
      typename std::conditional<(First == 1), Shape<>, Shape<First>>::type;
};

template <int64_t First, int64_t... Rest>
struct SqueezeShape<Shape<First, Rest...>> {
  using type = typename std::conditional<
      (First == 1), typename SqueezeShape<Shape<Rest...>>::type,
      typename SqueezeShape<Shape<Rest...>>::type::template prepend<First>>::
      type;
};

template <int64_t Idx, typename S>
struct SqueezeShapeAtIdx;

template <int64_t First, int64_t... Rest>
struct SqueezeShapeAtIdx<0, Shape<First, Rest...>> {
  using type = typename std::conditional<(First == 1), Shape<Rest...>,
                                         Shape<First, Rest...>>::type;
};

template <int64_t Idx, int64_t First, int64_t... Rest>
struct SqueezeShapeAtIdx<Idx, Shape<First, Rest...>> {
  static_assert(Idx < Shape<First, Rest...>::DIMS,
                "Idx must be smaller than the number of dimensions.");
  using type =
      typename SqueezeShapeAtIdx<Idx - 1,
                                 Shape<Rest...>>::type::template prepend<First>;
};

template <int64_t Curr, typename D, typename S>
struct VariadicSqueezeShape;

template <int64_t Curr, typename S>
struct VariadicSqueezeShape<Curr, Shape<>, S> {
  using type = S;
};

template <int64_t Curr, int64_t First, int64_t... Rest, typename S>
struct VariadicSqueezeShape<Curr, Shape<First, Rest...>, S> {
  static_assert(AreAllInIncreasingOrder<First, Rest...>::value,
                "The provided dimensions must be in increasing order.");
  static_assert(First - Curr >= 0 && First - Curr < S::DIMS,
                "Dimension out of bounds.");
  static_assert(S::SHAPE_DIMS[First - Curr] == 1,
                "Can only squeeze dimensions of size 1.");
  using type = typename VariadicSqueezeShape<
      Curr + 1, Shape<Rest...>,
      typename DropDimAt<First - Curr, S>::type>::type;
};

template <int64_t I, int64_t J, typename S>
struct SwapDimsShape {
  static_assert(I >= 0 && I < S::DIMS, "I must be smaller than Shape::DIMS.");
  static_assert(J >= 0 && J < S::DIMS, "J must be smaller than Shape::DIMS.");
  using type = typename SetDimAtIdx<
      J, S::SHAPE_DIMS[I],
      typename SetDimAtIdx<I, S::SHAPE_DIMS[J], S>::type>::type;
};

template <typename S, typename... RepeatPairs>
struct RepeatShape;

template <typename S>
struct RepeatShape<S> {
  using type = S;
};

template <typename S, typename First>
struct RepeatShape<S, First> {
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  using type =
      typename SetDimAtIdx<First::DIM, S::SHAPE_DIMS[First::DIM] * First::K,
                           S>::type;
};

template <typename S, typename First, typename... Rest>
struct RepeatShape<S, First, Rest...> {
  static_assert(AreTensorPairDimsUnique<First, Rest...>::value,
                "You cannot repeat the same dimension more than once.");
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  using type = typename RepeatShape<
      typename SetDimAtIdx<First::DIM, S::SHAPE_DIMS[First::DIM] * First::K,
                           S>::type,
      Rest...>::type;
};

template <typename S, typename... TopPairs>
struct TopShape;

template <typename S>
struct TopShape<S> {
  using type = S;
};

template <typename S, typename First>
struct TopShape<S, First> {
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  static_assert(First::K <= S::SHAPE_DIMS[First::DIM],
                "K is larger than the number of elements in that dimension.");
  using type = typename SetDimAtIdx<First::DIM, First::K, S>::type;
};

template <typename S, typename First, typename... Rest>
struct TopShape<S, First, Rest...> {
  static_assert(
      AreTensorPairDimsUnique<First, Rest...>::value,
      "You cannot take the top elements of the same dimension twice.");
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  static_assert(First::K <= S::SHAPE_DIMS[First::DIM],
                "K is larger than the number of elements in that dimension.");
  using type =
      typename TopShape<typename SetDimAtIdx<First::DIM, First::K, S>::type,
                        Rest...>::type;
};

template <typename S, typename... TensorRanges>
struct NarrowShape;

template <typename S>
struct NarrowShape<S> {
  using type = S;
};

template <typename S, typename First>
struct NarrowShape<S, First> {
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  static_assert(First::START <= First::END,
                "TensorRange::START must be <= TensorRange::END.");
  static_assert(First::END <= S::SHAPE_DIMS[First::DIM],
                "TensorRange::END is past the end of the dimension.");
  static_assert(First::START >= 0, "TensorRange::START must be >= 0.");
  using type =
      typename SetDimAtIdx<First::DIM, First::END - First::START, S>::type;
};

template <typename S, typename First, typename... Rest>
struct NarrowShape<S, First, Rest...> {
  static_assert(AreTensorPairDimsUnique<First, Rest...>::value,
                "You cannot narrow the same dimension more than once.");
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  static_assert(First::START <= First::END,
                "TensorRange::START must be <= TensorRange::END.");
  static_assert(First::END <= S::SHAPE_DIMS[First::DIM],
                "TensorRange::END is past the end of the dimension.");
  static_assert(First::START >= 0, "TensorRange::START must be >= 0.");
  using type = typename NarrowShape<
      typename SetDimAtIdx<First::DIM, First::END - First::START, S>::type,
      Rest...>::type;
};

// `slice` is `narrow`: it keeps the rank and shrinks the selected dimensions.
template <int64_t Curr, typename S, typename... TensorRanges>
struct SliceShape : NarrowShape<S, TensorRanges...> {};

template <typename S, typename... SelectPairs>
struct SelectOnlyShape;

template <typename S>
struct SelectOnlyShape<S> {
  using type = Shape<>;
};

template <typename S, typename First>
struct SelectOnlyShape<S, First> {
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  static_assert(AreAllLessThanN<S::SHAPE_DIMS[First::DIM],
                                typename First::Indices>::value,
                "All indices must be smaller than the selected dimension.");
  static_assert(AreAllUnique<typename First::Indices>::value,
                "You cannot select the same index twice.");
  using type = Shape<First::Indices::DIMS>;
};

template <typename S, typename First, typename... Rest>
struct SelectOnlyShape<S, First, Rest...> {
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  static_assert(AreAllLessThanN<S::SHAPE_DIMS[First::DIM],
                                typename First::Indices>::value,
                "All indices must be smaller than the selected dimension.");
  static_assert(AreAllUnique<typename First::Indices>::value,
                "You cannot select the same index twice.");
  static_assert(AreTensorPairDimsUnique<First, Rest...>::value,
                "You cannot select from the same dimension twice.");
  static_assert(AreTensorRangesAllInIncreasingOrder<First, Rest...>::value,
                "SelectPairs must be in increasing dimension order.");
  using type = typename SelectOnlyShape<S, Rest...>::type::template prepend<
      First::Indices::DIMS>;
};

template <typename S, typename... SelectPairs>
struct SelectShape;

template <typename S>
struct SelectShape<S> {
  using type = S;
};

template <typename S, typename First>
struct SelectShape<S, First> {
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  static_assert(AreAllLessThanN<S::SHAPE_DIMS[First::DIM],
                                typename First::Indices>::value,
                "All indices must be smaller than the selected dimension.");
  static_assert(AreAllUnique<typename First::Indices>::value,
                "You cannot select the same index twice.");
  using type = typename SetDimAtIdx<First::DIM, First::Indices::DIMS, S>::type;
};

template <typename S, typename First, typename... Rest>
struct SelectShape<S, First, Rest...> {
  static_assert(First::DIM >= 0 && First::DIM < S::DIMS,
                "Index out of bounds.");
  static_assert(AreAllLessThanN<S::SHAPE_DIMS[First::DIM],
                                typename First::Indices>::value,
                "All indices must be smaller than the selected dimension.");
  static_assert(AreAllUnique<typename First::Indices>::value,
                "You cannot select the same index twice.");
  static_assert(AreTensorPairDimsUnique<First, Rest...>::value,
                "You cannot select from the same dimension twice.");
  static_assert(AreTensorRangesAllInIncreasingOrder<First, Rest...>::value,
                "SelectPairs must be in increasing dimension order.");
  using type = typename SelectShape<
      typename SetDimAtIdx<First::DIM, First::Indices::DIMS, S>::type,
      Rest...>::type;
};

template <typename S, int64_t Height, int64_t Width>
struct PadShape {
  static_assert(S::DIMS >= 2,
                "Pad only works on tensors with at least 2 dimensions.");
  using type = typename SetDimAtIdx<
      S::DIMS - 1, S::SHAPE_DIMS[S::DIMS - 1] + Width,
      typename SetDimAtIdx<S::DIMS - 2, S::SHAPE_DIMS[S::DIMS - 2] + Height,
                           S>::type>::type;
};

///
///
/// TENSOR
///
///

template <typename TShape, typename TType = f32, typename TDevice = CPU<0>>
class Tensor {
 private:
  torch::Tensor impl_;

  explicit Tensor(torch::Tensor impl) : impl_(std::move(impl)) {}

 public:
  using TensorShape = TShape;
  using TensorType = dtype::DType<TType>;
  using TensorDevice = TDevice;

  static constexpr int64_t DIMS = TShape::DIMS;
  static constexpr int64_t NELEMS = TShape::NELEMS;
  inline static constexpr std::array<int64_t, DIMS> SHAPE_DIMS =
      TShape::SHAPE_DIMS;

  static constexpr torch::ScalarType DTYPE = TensorType::DTYPE;
  static constexpr size_t DTYPE_SIZE = TensorType::SIZE;

  static constexpr torch::DeviceType DEVICE = TDevice::DEVICE;
  static constexpr int64_t DEVICE_IDX = TDevice::INDEX;

  static torch::TensorOptions options() {
    return torch::TensorOptions().dtype(DTYPE).device(DEVICE, DEVICE_IDX);
  }

  // Constructors / assignment

  Tensor() : impl_(torch::zeros(SHAPE_DIMS, options())) {}

  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor& operator=(Tensor&&) noexcept = default;
  ~Tensor() = default;

  // Escape hatches

  [[nodiscard]] const torch::Tensor& native() const noexcept { return impl_; }
  [[nodiscard]] torch::Tensor& native() noexcept { return impl_; }
  [[nodiscard]] torch::Tensor to_torch() const { return impl_; }

  [[nodiscard]] bool defined() const noexcept { return impl_.defined(); }
  [[nodiscard]] int64_t dim() const noexcept { return impl_.dim(); }
  [[nodiscard]] int64_t numel() const noexcept { return impl_.numel(); }
  [[nodiscard]] std::vector<int64_t> sizes() const {
    return impl_.sizes().vec();
  }
  [[nodiscard]] torch::Device device() const { return impl_.device(); }
  [[nodiscard]] torch::ScalarType scalar_type() const {
    return impl_.scalar_type();
  }

  template <typename T>
  [[nodiscard]] T* data_ptr() const {
    return impl_.data_ptr<T>();
  }

  static Tensor from_torch_unchecked(torch::Tensor other) {
    return Tensor(std::move(other));
  }

  static Tensor from_torch(const torch::Tensor& other) {
    TORCH_CHECK(other.defined(),
                "incin::Tensor: cannot wrap an undefined tensor");
    TORCH_CHECK(
        other.sizes() == at::IntArrayRef(SHAPE_DIMS.data(), SHAPE_DIMS.size()),
        "incin::Tensor: shape mismatch");
    TORCH_CHECK(other.scalar_type() == DTYPE, "incin::Tensor: dtype mismatch");
    TORCH_CHECK(other.device().type() == DEVICE,
                "incin::Tensor: device type mismatch");
    if (DEVICE != torch::kCPU && DEVICE != torch::kMeta) {
      TORCH_CHECK(other.device().index() == DEVICE_IDX,
                  "incin::Tensor: device index mismatch");
    }
    return Tensor(other);
  }

  // Generators

  [[nodiscard]] static Tensor zeros() { return Tensor(); }

  [[nodiscard]] static Tensor ones() {
    return Tensor(torch::ones(SHAPE_DIMS, options()));
  }

  [[nodiscard]] static Tensor full(double value) {
    return Tensor(torch::full(SHAPE_DIMS, value, options()));
  }

  [[nodiscard]] static Tensor rand() {
    return Tensor(torch::rand(SHAPE_DIMS, options()));
  }

  [[nodiscard]] static Tensor randn() {
    return Tensor(torch::randn(SHAPE_DIMS, options()));
  }

  [[nodiscard]] Tensor zeros_like() const {
    return Tensor(torch::zeros_like(impl_));
  }

  [[nodiscard]] Tensor ones_like() const {
    return Tensor(torch::ones_like(impl_));
  }

  [[nodiscard]] Tensor full_like(double value) const {
    return Tensor(torch::full_like(impl_, value));
  }

  template <int64_t Start = 0, int64_t End, int64_t Step = 1>
  [[nodiscard]] static auto arange()
      -> Tensor<Shape<(End - Start) / Step>, TType, TDevice> {
    static_assert(Step > 0, "Step must be positive.");
    static_assert(Start <= End, "Start must be <= End.");
    static_assert((End - Start) % Step == 0,
                  "The range must be evenly divisible by Step.");
    static_assert(TShape::DIMS == 1,
                  "arange only creates one-dimensional tensors.");
    static_assert(TShape::NELEMS == (End - Start) / Step,
                  "Shape mismatch: the shape must match the arange length.");
    using Result = Tensor<Shape<(End - Start) / Step>, TType, TDevice>;
    return Result::from_torch_unchecked(
        torch::arange(Start, End, Step, options()));
  }

  // Operators

  [[nodiscard]] Tensor operator+(const Tensor& other) const {
    return Tensor(impl_ + other.impl_);
  }
  Tensor& operator+=(const Tensor& other) {
    impl_ += other.impl_;
    return *this;
  }

  [[nodiscard]] Tensor operator-(const Tensor& other) const {
    return Tensor(impl_ - other.impl_);
  }
  Tensor& operator-=(const Tensor& other) {
    impl_ -= other.impl_;
    return *this;
  }

  [[nodiscard]] Tensor operator*(const Tensor& other) const {
    return Tensor(impl_ * other.impl_);
  }
  Tensor& operator*=(const Tensor& other) {
    impl_ *= other.impl_;
    return *this;
  }

  [[nodiscard]] Tensor operator/(const Tensor& other) const {
    return Tensor(impl_ / other.impl_);
  }
  Tensor& operator/=(const Tensor& other) {
    impl_ /= other.impl_;
    return *this;
  }

  [[nodiscard]] Tensor operator-() const { return Tensor(-impl_); }

#define INCIN_DEF_SCALAR_OP(op)                                        \
  template <typename Scalar,                                           \
            typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>> \
  [[nodiscard]] Tensor operator op(Scalar scalar) const {              \
    return Tensor(impl_ op scalar);                                    \
  }                                                                    \
  template <typename Scalar,                                           \
            typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>> \
  Tensor& operator op##=(Scalar scalar) {                              \
    impl_ op## = scalar;                                               \
    return *this;                                                      \
  }

  INCIN_DEF_SCALAR_OP(+)
  INCIN_DEF_SCALAR_OP(-)
  INCIN_DEF_SCALAR_OP(*)
  INCIN_DEF_SCALAR_OP(/)

#undef INCIN_DEF_SCALAR_OP

  [[nodiscard]] bool operator==(const Tensor& other) const {
    return impl_.equal(other.impl_);
  }
  [[nodiscard]] bool operator!=(const Tensor& other) const {
    return !(*this == other);
  }

  [[nodiscard]] bool allclose(const Tensor& other, double rtol = 1e-5,
                              double atol = 1e-8) const {
    return torch::allclose(impl_, other.impl_, rtol, atol);
  }

  // Elementwise math

  [[nodiscard]] Tensor abs() const { return Tensor(impl_.abs()); }
  [[nodiscard]] Tensor exp() const { return Tensor(impl_.exp()); }
  [[nodiscard]] Tensor log() const { return Tensor(impl_.log()); }
  [[nodiscard]] Tensor sqrt() const { return Tensor(impl_.sqrt()); }
  [[nodiscard]] Tensor relu() const { return Tensor(impl_.relu()); }
  [[nodiscard]] Tensor sigmoid() const { return Tensor(impl_.sigmoid()); }
  [[nodiscard]] Tensor tanh() const { return Tensor(impl_.tanh()); }
  [[nodiscard]] Tensor pow(double exponent) const {
    return Tensor(impl_.pow(exponent));
  }
  [[nodiscard]] Tensor clamp(double min, double max) const {
    return Tensor(impl_.clamp(min, max));
  }
  [[nodiscard]] Tensor softmax(int64_t dim) const {
    return Tensor(impl_.softmax(dim));
  }
  [[nodiscard]] auto logical_not() const -> Tensor<TShape, bool, TDevice> {
    return Tensor<TShape, bool, TDevice>::from_torch_unchecked(
        impl_.logical_not());
  }

  // Reductions (return scalar tensors)

  [[nodiscard]] auto sum() const -> Tensor<Shape<>, TType, TDevice> {
    return Tensor<Shape<>, TType, TDevice>::from_torch_unchecked(impl_.sum());
  }
  [[nodiscard]] auto mean() const -> Tensor<Shape<>, TType, TDevice> {
    return Tensor<Shape<>, TType, TDevice>::from_torch_unchecked(impl_.mean());
  }
  [[nodiscard]] auto max() const -> Tensor<Shape<>, TType, TDevice> {
    return Tensor<Shape<>, TType, TDevice>::from_torch_unchecked(impl_.max());
  }
  [[nodiscard]] auto min() const -> Tensor<Shape<>, TType, TDevice> {
    return Tensor<Shape<>, TType, TDevice>::from_torch_unchecked(impl_.min());
  }

  template <typename T = double>
  [[nodiscard]] T item() const {
    static_assert(TShape::NELEMS == 1,
                  "item() is only available on single-element tensors.");
    return impl_.item<T>();
  }

  [[nodiscard]] Tensor clone() const { return Tensor(impl_.clone()); }
  [[nodiscard]] Tensor detach() const { return Tensor(impl_.detach()); }

  void print() const { std::cout << impl_ << std::endl; }

  // Reshape

  template <int64_t... NewDims>
  [[nodiscard]] auto reshape() const
      -> Tensor<Shape<NewDims...>, TType, TDevice> {
    static_assert(Shape<NewDims...>::NELEMS == TShape::NELEMS,
                  "Shape mismatch: reshape must preserve the number of "
                  "elements.");
    using Result = Tensor<Shape<NewDims...>, TType, TDevice>;
    return Result::from_torch_unchecked(
        impl_.reshape(ShapeToVector<Shape<NewDims...>>::get()));
  }

  template <typename NewShape>
  [[nodiscard]] auto reshape() const -> Tensor<NewShape, TType, TDevice> {
    static_assert(NewShape::NELEMS == TShape::NELEMS,
                  "Shape mismatch: reshape must preserve the number of "
                  "elements.");
    using Result = Tensor<NewShape, TType, TDevice>;
    return Result::from_torch_unchecked(
        impl_.reshape(ShapeToVector<NewShape>::get()));
  }

  // Matmul

  template <typename OtherShape>
  [[nodiscard]] auto matmul(const Tensor<OtherShape, TType, TDevice>& other)
      const -> Tensor<typename MatmulShape<TShape, OtherShape>::type, TType,
                      TDevice> {
    using Result =
        Tensor<typename MatmulShape<TShape, OtherShape>::type, TType, TDevice>;
    return Result::from_torch_unchecked(impl_.matmul(other.native()));
  }

  // Stack

  template <int64_t Dim = 0, size_t N>
  [[nodiscard]] static auto stack(const std::array<Tensor, N>& others)
      -> Tensor<
          typename InsertDimShape<Dim, static_cast<int64_t>(N), TShape>::type,
          TType, TDevice> {
    static_assert(Dim >= 0 && Dim <= TShape::DIMS, "Dimension out of bounds.");
    std::vector<torch::Tensor> tensors;
    tensors.reserve(N);
    for (const auto& tensor : others) {
      tensors.push_back(tensor.impl_);
    }
    using Result = Tensor<
        typename InsertDimShape<Dim, static_cast<int64_t>(N), TShape>::type,
        TType, TDevice>;
    return Result::from_torch_unchecked(torch::stack(tensors, Dim));
  }

  template <int64_t Dim = 0, size_t N>
  [[nodiscard]] static auto stack(const Tensor (&others)[N]) -> Tensor<
      typename InsertDimShape<Dim, static_cast<int64_t>(N), TShape>::type,
      TType, TDevice> {
    static_assert(Dim >= 0 && Dim <= TShape::DIMS, "Dimension out of bounds.");
    std::vector<torch::Tensor> tensors;
    tensors.reserve(N);
    for (size_t i = 0; i < N; ++i) {
      tensors.push_back(others[i].impl_);
    }
    using Result = Tensor<
        typename InsertDimShape<Dim, static_cast<int64_t>(N), TShape>::type,
        TType, TDevice>;
    return Result::from_torch_unchecked(torch::stack(tensors, Dim));
  }

  template <int64_t Dim = 0, typename First, typename Second, typename... Rest>
  [[nodiscard]] static auto stack(const First& first, const Second& second,
                                  const Rest&... rest)
      -> Tensor<typename InsertDimShape<Dim, 2 + sizeof...(Rest), TShape>::type,
                TType, TDevice> {
    static_assert(Dim >= 0 && Dim <= TShape::DIMS, "Dimension out of bounds.");
    static_assert(AreAllSameAs<Tensor, First, Second, Rest...>::value,
                  "All stacked tensors must have the same type.");
    std::vector<torch::Tensor> tensors{first.impl_, second.impl_};
    tensors.reserve(2 + sizeof...(Rest));
    (tensors.push_back(rest.impl_), ...);
    using Result =
        Tensor<typename InsertDimShape<Dim, 2 + sizeof...(Rest), TShape>::type,
               TType, TDevice>;
    return Result::from_torch_unchecked(torch::stack(tensors, Dim));
  }

  // Cat

  template <int64_t Dim = 0, size_t N>
  [[nodiscard]] static auto cat(const std::array<Tensor, N>& others) -> Tensor<
      typename CatArrayShape<TShape, Dim, static_cast<int64_t>(N)>::type, TType,
      TDevice> {
    static_assert(Dim >= 0 && Dim < TShape::DIMS, "Dimension out of bounds.");
    std::vector<torch::Tensor> tensors;
    tensors.reserve(N);
    for (const auto& tensor : others) {
      tensors.push_back(tensor.impl_);
    }
    using Result = Tensor<
        typename CatArrayShape<TShape, Dim, static_cast<int64_t>(N)>::type,
        TType, TDevice>;
    return Result::from_torch_unchecked(torch::cat(tensors, Dim));
  }

  template <int64_t Dim = 0, size_t N>
  [[nodiscard]] static auto cat(const Tensor (&others)[N]) -> Tensor<
      typename CatArrayShape<TShape, Dim, static_cast<int64_t>(N)>::type, TType,
      TDevice> {
    static_assert(Dim >= 0 && Dim < TShape::DIMS, "Dimension out of bounds.");
    std::vector<torch::Tensor> tensors;
    tensors.reserve(N);
    for (size_t i = 0; i < N; ++i) {
      tensors.push_back(others[i].impl_);
    }
    using Result = Tensor<
        typename CatArrayShape<TShape, Dim, static_cast<int64_t>(N)>::type,
        TType, TDevice>;
    return Result::from_torch_unchecked(torch::cat(tensors, Dim));
  }

  template <int64_t Dim = 0, typename... OtherShapes>
  [[nodiscard]] auto cat(const Tensor<OtherShapes, TType, TDevice>&... others)
      const -> Tensor<typename CatShape<Dim, TShape, OtherShapes...>::type,
                      TType, TDevice> {
    static_assert(Dim >= 0 && Dim < TShape::DIMS, "Dimension out of bounds.");
    std::vector<torch::Tensor> tensors{impl_};
    tensors.reserve(1 + sizeof...(OtherShapes));
    (tensors.push_back(others.native()), ...);
    using Result = Tensor<typename CatShape<Dim, TShape, OtherShapes...>::type,
                          TType, TDevice>;
    return Result::from_torch_unchecked(torch::cat(tensors, Dim));
  }

  // Transpose / permute

  template <int64_t... Dims>
  [[nodiscard]] auto transpose() const
      -> Tensor<typename TransposeShape<Shape<Dims...>, TShape>::type, TType,
                TDevice> {
    static_assert(sizeof...(Dims) == TShape::DIMS,
                  "The permutation must cover every dimension.");
    static_assert(AreAllUnique<Shape<Dims...>>::value,
                  "The permutation cannot contain duplicates.");
    static_assert(AreAllLessThanN<TShape::DIMS, Shape<Dims...>>::value,
                  "The permutation contains an out-of-range dimension.");
    using Result = Tensor<typename TransposeShape<Shape<Dims...>, TShape>::type,
                          TType, TDevice>;
    return Result::from_torch_unchecked(
        impl_.permute(ShapeToVector<Shape<Dims...>>::get()));
  }

  [[nodiscard]] auto flatten() const
      -> Tensor<Shape<TShape::NELEMS>, TType, TDevice> {
    return Tensor<Shape<TShape::NELEMS>, TType, TDevice>::from_torch_unchecked(
        impl_.flatten());
  }

  // Unsqueeze

  // Dimensions refer to the original shape and must be in increasing order.
  template <int64_t... Dims>
  [[nodiscard]] auto unsqueeze() const -> Tensor<
      typename VariadicUnsqueezeShape<0, Shape<Dims...>, TShape>::type, TType,
      TDevice> {
    using Result =
        Tensor<typename VariadicUnsqueezeShape<0, Shape<Dims...>, TShape>::type,
               TType, TDevice>;
    torch::Tensor out = impl_;
    int64_t offset = 0;
    ((out = out.unsqueeze(Dims + offset), ++offset), ...);
    return Result::from_torch_unchecked(out);
  }

  [[nodiscard]] auto unsqueeze() const
      -> Tensor<typename UnsqueezeShape<0, TShape>::type, TType, TDevice> {
    return this->template unsqueeze<0>();
  }

  template <int64_t Dim = 0>
  [[nodiscard]] auto unsqueeze_dim() const
      -> Tensor<typename UnsqueezeDimShape<Dim, TShape>::type, TType, TDevice> {
    static_assert(Dim >= 0 && Dim <= TShape::DIMS, "Dimension out of bounds.");
    using Result =
        Tensor<typename UnsqueezeDimShape<Dim, TShape>::type, TType, TDevice>;
    return Result::from_torch_unchecked(impl_.unsqueeze(Dim));
  }

  // Squeeze

  [[nodiscard]] auto squeeze() const
      -> Tensor<typename SqueezeShape<TShape>::type, TType, TDevice> {
    using Result = Tensor<typename SqueezeShape<TShape>::type, TType, TDevice>;
    return Result::from_torch_unchecked(impl_.squeeze());
  }

  template <int64_t... Dims>
  [[nodiscard]] auto squeeze() const
      -> Tensor<typename VariadicSqueezeShape<0, Shape<Dims...>, TShape>::type,
                TType, TDevice> {
    using Result =
        Tensor<typename VariadicSqueezeShape<0, Shape<Dims...>, TShape>::type,
               TType, TDevice>;
    torch::Tensor out = impl_;
    std::array<int64_t, sizeof...(Dims)> dims{Dims...};
    for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
      out = out.squeeze(*it);
    }
    return Result::from_torch_unchecked(out);
  }

  // Swap dims

  template <int64_t I = 0, int64_t J = 1>
  [[nodiscard]] auto swap_dims() const
      -> Tensor<typename SwapDimsShape<I, J, TShape>::type, TType, TDevice> {
    using Result =
        Tensor<typename SwapDimsShape<I, J, TShape>::type, TType, TDevice>;
    return Result::from_torch_unchecked(impl_.transpose(I, J));
  }

  // Permute

  template <int64_t... Dims>
  [[nodiscard]] auto permute() const
      -> Tensor<typename TransposeShape<Shape<Dims...>, TShape>::type, TType,
                TDevice> {
    static_assert(sizeof...(Dims) == TShape::DIMS,
                  "The permutation must cover every dimension.");
    static_assert(AreAllUnique<Shape<Dims...>>::value,
                  "The permutation cannot contain duplicates.");
    static_assert(AreAllLessThanN<TShape::DIMS, Shape<Dims...>>::value,
                  "The permutation contains an out-of-range dimension.");
    using Result = Tensor<typename TransposeShape<Shape<Dims...>, TShape>::type,
                          TType, TDevice>;
    return Result::from_torch_unchecked(
        impl_.permute(ShapeToVector<Shape<Dims...>>::get()));
  }

  // Repeat

  template <typename... RepeatPairs>
  [[nodiscard]] auto repeat() const
      -> Tensor<typename RepeatShape<TShape, RepeatPairs...>::type, TType,
                TDevice> {
    std::vector<int64_t> repeats(TShape::DIMS, 1);
    ((repeats[RepeatPairs::DIM] = RepeatPairs::K), ...);
    using Result = Tensor<typename RepeatShape<TShape, RepeatPairs...>::type,
                          TType, TDevice>;
    return Result::from_torch_unchecked(impl_.repeat(repeats));
  }

  // Narrow (a.k.a. top / slice) from the start of each dimension.

  template <typename... TopPairs>
  [[nodiscard]] auto top() const
      -> Tensor<typename TopShape<TShape, TopPairs...>::type, TType, TDevice> {
    torch::Tensor out = impl_;
    ((out = out.narrow(TopPairs::DIM, 0, TopPairs::K)), ...);
    using Result =
        Tensor<typename TopShape<TShape, TopPairs...>::type, TType, TDevice>;
    return Result::from_torch_unchecked(out);
  }

  // Apply

  void apply(const std::function<void(Tensor*)>& fn) { fn(this); }
  void apply(const std::function<void(const Tensor*)>& fn) const { fn(this); }

  // Narrow

  template <typename... TensorRanges>
  [[nodiscard]] auto narrow() const
      -> Tensor<typename NarrowShape<TShape, TensorRanges...>::type, TType,
                TDevice> {
    torch::Tensor out = impl_;
    ((out = out.narrow(TensorRanges::DIM, TensorRanges::START,
                       TensorRanges::END - TensorRanges::START)),
     ...);
    using Result = Tensor<typename NarrowShape<TShape, TensorRanges...>::type,
                          TType, TDevice>;
    return Result::from_torch_unchecked(out);
  }

  template <int64_t Start = 0, int64_t End, int64_t Dim = 0>
  [[nodiscard]] auto narrow() const
      -> Tensor<typename NarrowShape<TShape, Tr<Start, End, Dim>>::type, TType,
                TDevice> {
    return this->template narrow<Tr<Start, End, Dim>>();
  }

  // Slice (keeps rank; same semantics as narrow)

  template <typename... TensorRanges>
  [[nodiscard]] auto slice() const
      -> Tensor<typename NarrowShape<TShape, TensorRanges...>::type, TType,
                TDevice> {
    torch::Tensor out = impl_;
    ((out = out.narrow(TensorRanges::DIM, TensorRanges::START,
                       TensorRanges::END - TensorRanges::START)),
     ...);
    using Result = Tensor<typename NarrowShape<TShape, TensorRanges...>::type,
                          TType, TDevice>;
    return Result::from_torch_unchecked(out);
  }

  template <int64_t Start = 0, int64_t End, int64_t Dim = 0>
  [[nodiscard]] auto slice() const
      -> Tensor<typename NarrowShape<TShape, Tr<Start, End, Dim>>::type, TType,
                TDevice> {
    return this->template slice<Tr<Start, End, Dim>>();
  }

  // dtype / device

  template <typename DType = f32>
  [[nodiscard]] auto to_dtype(bool async = false) const
      -> Tensor<TShape, DType, TDevice> {
    return Tensor<TShape, DType, TDevice>::from_torch_unchecked(
        impl_.to(dtype::DType<DType>::DTYPE, async));
  }

  template <typename Device = CPU<0>>
  [[nodiscard]] auto to_device(bool async = false) const
      -> Tensor<TShape, TType, Device> {
    return Tensor<TShape, TType, Device>::from_torch_unchecked(impl_.to(
        torch::TensorOptions().device(Device::DEVICE, Device::INDEX), async));
  }

  // Chunk

  template <int64_t Chunks = 1, int64_t Dim = 0>
  [[nodiscard]] auto chunk() const -> std::array<
      Tensor<typename SetDimAtIdx<Dim, TShape::SHAPE_DIMS[Dim] / Chunks,
                                  TShape>::type,
             TType, TDevice>,
      Chunks> {
    static_assert(Chunks > 0, "Chunks must be positive.");
    static_assert(Dim >= 0 && Dim < TShape::DIMS, "Dimension out of bounds.");
    static_assert(TShape::SHAPE_DIMS[Dim] % Chunks == 0,
                  "The dimension must be evenly divisible by Chunks.");
    using Element =
        Tensor<typename SetDimAtIdx<Dim, TShape::SHAPE_DIMS[Dim] / Chunks,
                                    TShape>::type,
               TType, TDevice>;
    std::array<Element, static_cast<size_t>(Chunks)> result{};
    auto parts = impl_.chunk(Chunks, Dim);
    for (size_t i = 0; i < static_cast<size_t>(Chunks); ++i) {
      result[i] = Element::from_torch_unchecked(parts[static_cast<int64_t>(i)]);
    }
    return result;
  }

  // Select (index_select, keeping rank)

  template <typename Pair>
  static void index_select_in_place(torch::Tensor& out) {
    auto indices = torch::tensor(
        ShapeToVector<typename Pair::Indices>::get(),
        torch::TensorOptions().dtype(torch::kLong).device(out.device()));
    out = out.index_select(Pair::DIM, indices);
  }

  template <typename... SelectPairs>
  [[nodiscard]] auto select() const
      -> Tensor<typename SelectShape<TShape, SelectPairs...>::type, TType,
                TDevice> {
    torch::Tensor out = impl_;
    (index_select_in_place<SelectPairs>(out), ...);
    using Result = Tensor<typename SelectShape<TShape, SelectPairs...>::type,
                          TType, TDevice>;
    return Result::from_torch_unchecked(out);
  }

  template <int64_t Dim = 0, int64_t... Indices>
  [[nodiscard]] auto select() const
      -> Tensor<typename SelectShape<TShape, Ip<Dim, Indices...>>::type, TType,
                TDevice> {
    return this->template select<Ip<Dim, Indices...>>();
  }

  // Select only (keep the selected dimensions, drop the rest at index 0)

  template <typename... SelectPairs>
  [[nodiscard]] auto select_only() const
      -> Tensor<typename SelectOnlyShape<TShape, SelectPairs...>::type, TType,
                TDevice> {
    torch::Tensor out = impl_;
    (index_select_in_place<SelectPairs>(out), ...);
    std::array<int64_t, sizeof...(SelectPairs)> selected{SelectPairs::DIM...};
    for (int64_t d = TShape::DIMS - 1; d >= 0; --d) {
      if (std::find(selected.begin(), selected.end(), d) == selected.end()) {
        out = out.select(d, 0);
      }
    }
    using Result =
        Tensor<typename SelectOnlyShape<TShape, SelectPairs...>::type, TType,
               TDevice>;
    return Result::from_torch_unchecked(out);
  }

  template <int64_t Dim = 0, int64_t... Indices>
  [[nodiscard]] auto select_only() const
      -> Tensor<typename SelectOnlyShape<TShape, Ip<Dim, Indices...>>::type,
                TType, TDevice> {
    return this->template select_only<Ip<Dim, Indices...>>();
  }

  // Pad the last two dimensions.

  template <int64_t Top, int64_t Bottom, int64_t Left, int64_t Right,
            typename Scalar>
  [[nodiscard]] auto pad(Scalar value)
      -> Tensor<typename PadShape<TShape, Top + Bottom, Left + Right>::type,
                TType, TDevice> {
    using Result =
        Tensor<typename PadShape<TShape, Top + Bottom, Left + Right>::type,
               TType, TDevice>;
    return Result::from_torch_unchecked(
        torch::constant_pad_nd(impl_, {Left, Right, Top, Bottom}, value));
  }

  // Boolean helper

  template <typename T = TType>
  [[nodiscard]] std::enable_if_t<std::is_same_v<T, bool>, Tensor> bool_not()
      const {
    static_assert(std::is_same_v<T, TType>,
                  "bool_not is only available on boolean tensors.");
    return Tensor(impl_.logical_not());
  }
};

template <typename S, typename D, typename Dev>
std::ostream& operator<<(std::ostream& os, const Tensor<S, D, Dev>& tensor) {
  return os << tensor.native();
}

}  // namespace incin

#endif  // INCIN_TENSOR_HPP
