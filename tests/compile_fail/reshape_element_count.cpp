// Must not compile: reshape cannot change the number of elements.
#include <incin/incin.hpp>

int main() {
  using namespace incin;

  auto a = Tensor<Shape<2, 3>>::ones();
  auto b = a.reshape<4, 2>();
  (void)b;
}
