// Must not compile: a permutation cannot repeat a dimension.
#include <incin/incin.hpp>

int main() {
  using namespace incin;

  auto a = Tensor<Shape<2, 3>>::ones();
  auto b = a.transpose<0, 0>();
  (void)b;
}
