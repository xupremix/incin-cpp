// Numerical and shape tests for incin-cpp.
//
// The compile-time shape checks are exercised simply by compiling this file.
// The runtime checks below make sure the surviving programs produce the values
// libtorch would produce for the same operation.

#include <cmath>
#include <cstdlib>
#include <incin/incin.hpp>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

using namespace incin;

namespace {

int g_failures = 0;

void fail(const char* expr, const char* file, int line) {
  std::cerr << "FAIL " << file << ":" << line << ": " << expr << '\n';
  ++g_failures;
}

#define INCIN_CHECK(expr)              \
  do {                                 \
    if (!(expr)) {                     \
      fail(#expr, __FILE__, __LINE__); \
    }                                  \
  } while (false)

#define INCIN_CHECK_CLOSE(actual, expected, tolerance)                       \
  do {                                                                       \
    const double a_ = static_cast<double>(actual);                           \
    const double e_ = static_cast<double>(expected);                         \
    if (std::abs(a_ - e_) > (tolerance)) {                                   \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #actual \
                << " = " << a_ << ", expected " << e_ << '\n';               \
      ++g_failures;                                                          \
    }                                                                        \
  } while (false)

#define INCIN_CHECK_SHAPE(tensor, ...)                                   \
  do {                                                                   \
    const std::vector<int64_t> expected_{__VA_ARGS__};                   \
    if ((tensor).sizes() != expected_) {                                 \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": shape "; \
      for (auto d : (tensor).sizes()) std::cerr << d << ' ';             \
      std::cerr << "!= ";                                                \
      for (auto d : expected_) std::cerr << d << ' ';                    \
      std::cerr << '\n';                                                 \
      ++g_failures;                                                      \
    }                                                                    \
  } while (false)

using M22 = Tensor<Shape<2, 2>>;
using M33 = Tensor<Shape<3, 3>>;

M22 mat22() {
  return M22::from_torch(
      torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, torch::kFloat32));
}

M33 mat33() {
  return M33::from_torch(torch::arange(9, torch::kFloat32).reshape({3, 3}));
}

void test_generators() {
  using T = Tensor<Shape<2, 3>>;

  INCIN_CHECK(torch::equal(T::zeros().to_torch(), torch::zeros({2, 3})));
  INCIN_CHECK(torch::equal(T::ones().to_torch(), torch::ones({2, 3})));
  INCIN_CHECK(torch::equal(T::full(2.5).to_torch(), torch::full({2, 3}, 2.5)));

  const auto random = T::rand();
  INCIN_CHECK_SHAPE(random, 2, 3);
  INCIN_CHECK_SHAPE(T::randn(), 2, 3);

  const auto scalar = Tensor<Shape<>>::full(42.0);
  INCIN_CHECK(scalar.sizes().empty());
  INCIN_CHECK_CLOSE(scalar.item<float>(), 42.0f, 1e-6);

  const auto like = T::ones().zeros_like();
  INCIN_CHECK(torch::equal(like.to_torch(), torch::zeros({2, 3})));
  INCIN_CHECK(
      torch::equal(T::zeros().ones_like().to_torch(), torch::ones({2, 3})));
  INCIN_CHECK(torch::equal(T::zeros().full_like(7.0).to_torch(),
                           torch::full({2, 3}, 7.0)));
}

void test_arithmetic() {
  const auto a = mat22();
  const auto b = M22::from_torch(
      torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}}, torch::kFloat32));

  INCIN_CHECK(torch::equal(
      (a + b).to_torch(),
      torch::tensor({{6.0f, 8.0f}, {10.0f, 12.0f}}, torch::kFloat32)));
  INCIN_CHECK(torch::equal(
      (b - a).to_torch(),
      torch::tensor({{4.0f, 4.0f}, {4.0f, 4.0f}}, torch::kFloat32)));
  INCIN_CHECK(torch::equal(
      (a * b).to_torch(),
      torch::tensor({{5.0f, 12.0f}, {21.0f, 32.0f}}, torch::kFloat32)));
  INCIN_CHECK(torch::allclose((b / a).to_torch(), b.to_torch() / a.to_torch()));
  INCIN_CHECK(torch::equal(
      (-a).to_torch(),
      torch::tensor({{-1.0f, -2.0f}, {-3.0f, -4.0f}}, torch::kFloat32)));

  auto c = mat22();
  c += b;
  INCIN_CHECK(torch::equal(c.to_torch(), (a + b).to_torch()));

  INCIN_CHECK(torch::equal((a + 1.0).to_torch(), a.to_torch() + 1.0));
  INCIN_CHECK(torch::equal((a * 2.0).to_torch(), a.to_torch() * 2.0));
}

void test_matmul() {
  using A = Tensor<Shape<1, 2, 3>>;
  using B = Tensor<Shape<1, 3, 4>>;

  const auto a = A::ones();
  const auto b = B::ones();
  const auto c = a.matmul(b);

  static_assert(std::is_same_v<decltype(c), const Tensor<Shape<1, 2, 4>>>);
  INCIN_CHECK_SHAPE(c, 1, 2, 4);
  INCIN_CHECK(torch::allclose(c.to_torch(), torch::full({1, 2, 4}, 3.0f)));

  const auto left = M22::from_torch(
      torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, torch::kFloat32));
  const auto right = M22::from_torch(
      torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}}, torch::kFloat32));
  INCIN_CHECK(torch::equal((left.matmul(right)).to_torch(),
                           left.to_torch().matmul(right.to_torch())));
}

void test_reshape() {
  const auto a = mat22();

  const auto flat = a.reshape<4, 1>();
  static_assert(std::is_same_v<decltype(flat), const Tensor<Shape<4, 1>>>);
  INCIN_CHECK_SHAPE(flat, 4, 1);
  INCIN_CHECK(torch::equal(
      flat.to_torch(),
      torch::tensor({{1.0f}, {2.0f}, {3.0f}, {4.0f}}, torch::kFloat32)));

  const auto row = a.reshape<Shape<1, 4>>();
  INCIN_CHECK_SHAPE(row, 1, 4);
  INCIN_CHECK(torch::equal(row.to_torch(), a.to_torch().reshape({1, 4})));
}

void test_transpose_permute() {
  const auto a = mat22();

  const auto transposed = a.transpose<1, 0>();
  INCIN_CHECK(torch::equal(
      transposed.to_torch(),
      torch::tensor({{1.0f, 3.0f}, {2.0f, 4.0f}}, torch::kFloat32)));

  const auto swapped = a.swap_dims<0, 1>();
  INCIN_CHECK(torch::equal(swapped.to_torch(), transposed.to_torch()));

  using T3 = Tensor<Shape<2, 3, 4>>;
  const auto t3 =
      T3::from_torch(torch::arange(24, torch::kFloat32).reshape({2, 3, 4}));
  const auto permuted = t3.permute<2, 0, 1>();
  static_assert(
      std::is_same_v<decltype(permuted), const Tensor<Shape<4, 2, 3>>>);
  INCIN_CHECK_SHAPE(permuted, 4, 2, 3);
  INCIN_CHECK(
      torch::equal(permuted.to_torch(), t3.to_torch().permute({2, 0, 1})));
}

void test_flatten() {
  const auto flat = mat22().flatten();
  static_assert(std::is_same_v<decltype(flat), const Tensor<Shape<4>>>);
  INCIN_CHECK(
      torch::equal(flat.to_torch(),
                   torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat32)));
}

void test_unsqueeze_squeeze() {
  const auto a = mat22();

  // Dimensions refer to the original shape and must be in increasing order.
  const auto upper = a.unsqueeze<0, 2>();
  static_assert(
      std::is_same_v<decltype(upper), const Tensor<Shape<1, 2, 2, 1>>>);
  INCIN_CHECK_SHAPE(upper, 1, 2, 2, 1);
  INCIN_CHECK(
      torch::equal(upper.to_torch(), a.to_torch().unsqueeze(0).unsqueeze(3)));

  const auto back = upper.squeeze<0, 3>();
  static_assert(std::is_same_v<decltype(back), const Tensor<Shape<2, 2>>>);
  INCIN_CHECK(back == a);

  const auto middle = a.unsqueeze_dim<1>();
  static_assert(std::is_same_v<decltype(middle), const Tensor<Shape<2, 1, 2>>>);
  INCIN_CHECK_SHAPE(middle, 2, 1, 2);
  INCIN_CHECK(middle.squeeze() == a);

  const auto leading = a.unsqueeze();
  INCIN_CHECK_SHAPE(leading, 1, 2, 2);
  INCIN_CHECK(leading.squeeze() == a);
}

void test_stack_cat() {
  const auto a = mat22();
  const auto b = a * 2.0;

  const std::array<M22, 2> pair{a, b};
  const auto stacked = M22::stack<0>(pair);
  static_assert(
      std::is_same_v<decltype(stacked), const Tensor<Shape<2, 2, 2>>>);
  INCIN_CHECK_SHAPE(stacked, 2, 2, 2);
  INCIN_CHECK(torch::equal(stacked.to_torch().select(0, 1), b.to_torch()));

  const auto middle_stack = M22::stack<1>(a, b);
  static_assert(
      std::is_same_v<decltype(middle_stack), const Tensor<Shape<2, 2, 2>>>);
  INCIN_CHECK(torch::equal(middle_stack.to_torch().select(1, 0), a.to_torch()));

  const auto rows = a.cat(b);
  static_assert(std::is_same_v<decltype(rows), const Tensor<Shape<4, 2>>>);
  INCIN_CHECK_SHAPE(rows, 4, 2);
  INCIN_CHECK(
      torch::equal(rows.to_torch(), torch::cat({a.to_torch(), b.to_torch()})));

  const auto cols = a.cat<1>(b);
  static_assert(std::is_same_v<decltype(cols), const Tensor<Shape<2, 4>>>);
  INCIN_CHECK_SHAPE(cols, 2, 4);
}

void test_arange() {
  const auto r = Tensor<Shape<5>>::arange<0, 5>();
  INCIN_CHECK(torch::equal(r.to_torch(), torch::arange(5, torch::kFloat32)));

  const auto stepped = Tensor<Shape<3>>::arange<1, 7, 2>();
  INCIN_CHECK(torch::equal(stepped.to_torch(),
                           torch::tensor({1.0f, 3.0f, 5.0f}, torch::kFloat32)));
}

void test_repeat_top_narrow_slice() {
  const auto a = mat22();

  const auto repeated = a.repeat<Tp<0, 2>, Tp<1, 3>>();
  static_assert(std::is_same_v<decltype(repeated), const Tensor<Shape<4, 6>>>);
  INCIN_CHECK(torch::equal(repeated.to_torch(), a.to_torch().repeat({2, 3})));

  const auto top = mat33().top<Tp<0, 2>, Tp<1, 2>>();
  static_assert(std::is_same_v<decltype(top), const Tensor<Shape<2, 2>>>);
  INCIN_CHECK(torch::equal(
      top.to_torch(),
      torch::tensor({{0.0f, 1.0f}, {3.0f, 4.0f}}, torch::kFloat32)));

  const auto narrowed = mat33().narrow<Tr<1, 3, 0>>();
  static_assert(std::is_same_v<decltype(narrowed), const Tensor<Shape<2, 3>>>);
  INCIN_CHECK(torch::equal(
      narrowed.to_torch(),
      torch::arange(9, torch::kFloat32).reshape({3, 3}).narrow(0, 1, 2)));

  const auto sliced = mat33().slice<Tr<1, 3, 1>>();
  static_assert(std::is_same_v<decltype(sliced), const Tensor<Shape<3, 2>>>);
  INCIN_CHECK(torch::equal(
      sliced.to_torch(),
      torch::arange(9, torch::kFloat32).reshape({3, 3}).narrow(1, 1, 2)));

  const auto repeated_noop = a.repeat<>();
  static_assert(
      std::is_same_v<decltype(repeated_noop), const Tensor<Shape<2, 2>>>);
  INCIN_CHECK(repeated_noop == a);

  const auto top_noop = a.top<>();
  static_assert(std::is_same_v<decltype(top_noop), const Tensor<Shape<2, 2>>>);
  INCIN_CHECK(top_noop == a);

  const auto narrowed_noop = a.narrow<>();
  static_assert(
      std::is_same_v<decltype(narrowed_noop), const Tensor<Shape<2, 2>>>);
  INCIN_CHECK(narrowed_noop == a);
}

void test_select() {
  const auto m = mat33();
  const auto rows = m.select<0, 0, 2>();
  static_assert(std::is_same_v<decltype(rows), const Tensor<Shape<2, 3>>>);
  INCIN_CHECK(torch::equal(
      rows.to_torch(),
      m.to_torch().index_select(0, torch::tensor({0, 2}, torch::kLong))));

  const auto both = m.select<Ip<0, 0, 2>, Ip<1, 0, 1>>();
  static_assert(std::is_same_v<decltype(both), const Tensor<Shape<2, 2>>>);
  INCIN_CHECK(
      torch::equal(both.to_torch(),
                   m.to_torch()
                       .index_select(0, torch::tensor({0, 2}, torch::kLong))
                       .index_select(1, torch::tensor({0, 1}, torch::kLong))));

  const auto only = m.select_only<0, 0, 2>();
  static_assert(std::is_same_v<decltype(only), const Tensor<Shape<2>>>);
  INCIN_CHECK(torch::equal(only.to_torch(),
                           torch::tensor({0.0f, 6.0f}, torch::kFloat32)));

  const auto only_two = m.select_only<Ip<0, 0, 2>, Ip<1, 0, 2>>();
  static_assert(std::is_same_v<decltype(only_two), const Tensor<Shape<2, 2>>>);
  INCIN_CHECK(torch::equal(
      only_two.to_torch(),
      torch::tensor({{0.0f, 2.0f}, {6.0f, 8.0f}}, torch::kFloat32)));
}

void test_chunk() {
  using T4 = Tensor<Shape<4, 4>>;
  const auto m =
      T4::from_torch(torch::arange(16, torch::kFloat32).reshape({4, 4}));
  const auto parts = m.chunk<2, 0>();
  static_assert(std::is_same_v<decltype(parts),
                               const std::array<Tensor<Shape<2, 4>>, 2>>);
  INCIN_CHECK(torch::equal(parts[0].to_torch(), m.to_torch().narrow(0, 0, 2)));
  INCIN_CHECK(torch::equal(parts[1].to_torch(), m.to_torch().narrow(0, 2, 2)));
}

void test_pad() {
  const auto padded = mat22().pad<1, 1, 1, 1>(0.0);
  static_assert(std::is_same_v<decltype(padded), const Tensor<Shape<4, 4>>>);
  INCIN_CHECK(
      torch::equal(padded.to_torch(), torch::tensor({{0.0f, 0.0f, 0.0f, 0.0f},
                                                     {0.0f, 1.0f, 2.0f, 0.0f},
                                                     {0.0f, 3.0f, 4.0f, 0.0f},
                                                     {0.0f, 0.0f, 0.0f, 0.0f}},
                                                    torch::kFloat32)));
}

void test_dtype_device() {
  const auto a = mat22();

  const auto d = a.to_dtype<f64>();
  static_assert(std::is_same_v<decltype(d), const Tensor<Shape<2, 2>, f64>>);
  INCIN_CHECK(d.to_torch().scalar_type() == torch::kFloat64);
  INCIN_CHECK(torch::equal(d.to_torch(), a.to_torch().to(torch::kFloat64)));

  const auto cpu = a.to_device<CPU<0>>();
  INCIN_CHECK(cpu == a);
}

void test_bool() {
  using Boolean = Tensor<Shape<2>, bool>;
  const auto b =
      Boolean::from_torch(torch::tensor({true, false}, torch::kBool));
  INCIN_CHECK(torch::equal(b.bool_not().to_torch(),
                           torch::tensor({false, true}, torch::kBool)));
  INCIN_CHECK(torch::equal(b.logical_not().to_torch(),
                           torch::tensor({false, true}, torch::kBool)));
}

void test_math_and_item() {
  const auto a = mat22();

  INCIN_CHECK(torch::allclose(a.exp().to_torch(), a.to_torch().exp()));
  INCIN_CHECK(torch::allclose(a.log().to_torch(), a.to_torch().log()));
  INCIN_CHECK(torch::allclose(a.sqrt().to_torch(), a.to_torch().sqrt()));
  INCIN_CHECK(torch::allclose(a.relu().to_torch(), a.to_torch().relu()));
  INCIN_CHECK(torch::allclose(a.sigmoid().to_torch(), a.to_torch().sigmoid()));
  INCIN_CHECK(torch::allclose(a.tanh().to_torch(), a.to_torch().tanh()));
  INCIN_CHECK(torch::allclose(a.pow(2.0).to_torch(), a.to_torch().pow(2.0)));
  INCIN_CHECK(torch::allclose(a.clamp(2.0, 3.0).to_torch(),
                              a.to_torch().clamp(2.0, 3.0)));
  INCIN_CHECK(
      torch::allclose(a.softmax(1).to_torch(), a.to_torch().softmax(1)));

  INCIN_CHECK_CLOSE(a.sum().item<float>(), 10.0f, 1e-6);
  INCIN_CHECK_CLOSE(a.mean().item<float>(), 2.5f, 1e-6);
  INCIN_CHECK_CLOSE(a.max().item<float>(), 4.0f, 1e-6);
  INCIN_CHECK_CLOSE(a.min().item<float>(), 1.0f, 1e-6);
}

void test_cuda_roundtrip() {
  if (!torch::cuda::is_available()) {
    std::cout << "CUDA not available; skipping CUDA roundtrip\n";
    return;
  }

  const auto cpu = mat22();
  const auto gpu = cpu.to_device<CUDA<0>>();
  static_assert(
      std::is_same_v<decltype(gpu), const Tensor<Shape<2, 2>, f32, CUDA<0>>>);
  CHECK(gpu.device().type() == torch::kCUDA);
  CHECK(gpu.device().index() == 0);
  CHECK(torch::equal(gpu.to_device<CPU<0>>().to_torch(), cpu.to_torch()));

  using A = Tensor<Shape<2, 3>, f32, CUDA<0>>;
  using B = Tensor<Shape<3, 4>, f32, CUDA<0>>;
  const auto c = A::ones().matmul(B::ones());
  CHECK(c.device().type() == torch::kCUDA);
  INCIN_CHECK_SHAPE(c, 2, 4);
  CHECK(torch::allclose(c.to_device<CPU<0>>().to_torch(),
                        torch::full({2, 4}, 3.0f)));
}

void test_copy_move() {
  const auto a = mat22();

  auto copy = a;
  INCIN_CHECK(copy == a);

  auto moved = std::move(copy);
  INCIN_CHECK(moved == a);

  M22 assigned;
  assigned = a;
  INCIN_CHECK(assigned == a);

  M22 move_assigned;
  move_assigned = std::move(assigned);
  INCIN_CHECK(move_assigned == a);

  INCIN_CHECK(a.clone() == a);
  INCIN_CHECK(a.detach() == a);
}

void test_from_torch_validation() {
  bool threw = false;
  try {
    (void)M22::from_torch(torch::zeros({3, 3}));
  } catch (const c10::Error&) {
    threw = true;
  }
  INCIN_CHECK(threw);
}

}  // namespace

int main() {
  test_generators();
  test_arithmetic();
  test_matmul();
  test_reshape();
  test_transpose_permute();
  test_flatten();
  test_unsqueeze_squeeze();
  test_stack_cat();
  test_arange();
  test_repeat_top_narrow_slice();
  test_select();
  test_chunk();
  test_pad();
  test_dtype_device();
  test_bool();
  test_math_and_item();
  test_cuda_roundtrip();
  test_copy_move();
  test_from_torch_validation();

  if (g_failures != 0) {
    std::cerr << g_failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "all incin-cpp tests passed\n";
  return EXIT_SUCCESS;
}
