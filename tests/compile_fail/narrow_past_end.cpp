// Must not compile: the narrow range extends past the dimension.
#include <incin/incin.hpp>

int main() {
  using namespace incin;

  auto a = Tensor<Shape<2, 3>>::ones();
  auto b = a.narrow<0, 5>();
  (void)b;
}
