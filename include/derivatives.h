#ifndef cfl_derivatives_h
#define cfl_derivatives_h

#include <string>
#include <traits.h>

namespace CFL
{
template <class T>
class Gradient
{
  const T t;

public:
  typedef Traits::Tensor<T::Traits::rank + 1, T::Traits::dim> Traits;
  Traits traits;

  Gradient(const T& t)
    : t(t)
  {
  }

  template <typename... Comp>
  std::string
  latex(std::array<int, Traits::dim> derivatives, int d, Comp... comp) const
  {
    // Discuss whether this copy is necessary, or if we want to hand
    // over derivatives by reference and screw it up for the caller
    ++derivatives[d];
    return t.latex(derivatives, comp...);
  }

  template <typename... Comp>
  std::string
  latex(std::array<int, Traits::dim> derivatives) const
  {
    return std::string("\nabla ") + t.latex(derivatives);
  }
 };

template <class T>
Gradient<T>
grad(const T t)
{
  return Gradient<T>(t);
}
}

#endif
