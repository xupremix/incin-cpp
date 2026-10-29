// Must not compile: the declared shape disagrees with the arange length.
#include <incin/incin.hpp>

int main() {
  using namespace incin;

  auto values = Tensor<Shape<4>>::arange<0, 5>();
  (void)values;
}
