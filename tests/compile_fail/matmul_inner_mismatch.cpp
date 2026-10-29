// Must not compile: the inner dimensions of the matmul operands disagree.
#include <incin/incin.hpp>

int main() {
  using namespace incin;

  auto a = Tensor<Shape<2, 3>>::ones();
  auto b = Tensor<Shape<2, 4>>::ones();
  auto c = a.matmul(b);
  (void)c;
}
