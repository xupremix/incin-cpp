#include <incin/incin.hpp>

int main() {
  using namespace incin;

  auto a = Tensor<Shape<2, 3>>::ones();
  auto b = Tensor<Shape<3, 1>>::ones();
  auto c = a.matmul(b);

  return c.to_torch().sum().item<float>() == 6.0f ? 0 : 1;
}
