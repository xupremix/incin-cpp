// A compile-time-checked matmul.

#include <incin/incin.hpp>

using namespace incin;

int main() {
  auto t1 = Tensor<Shape<1, 2, 3>, f32, CPU<0>>::ones();
  auto t2 = Tensor<Shape<1, 3, 4>, f32, CPU<0>>::ones();

  auto t3 = t1.matmul(t2);  // [1, 2, 3] x [1, 3, 4] -> [1, 2, 4]

  t3.print();

  // The following would be a compile error:
  // auto bad = Tensor<Shape<2, 3>, f32, CPU<0>>::ones();
  // t1.matmul(bad);

  return 0;
}
