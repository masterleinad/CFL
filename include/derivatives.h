#ifndef cfl_derivatives_h
#define cfl_derivatives_h

#include <string>
#include <array>
#include <traits.h>

namespace CFL
{
/**
 * \brief The gradient as a tensor as a tensor of higher rank
 *
 * Given the parameter class T of rank <i>k</i>, this class is an
 * object of tensor rank <i>k</i>+1.
 *
 * \warning In access funtions, the derivative direction is the
 * first index in the resulting tensor, thus applying the transpose
 * of the notion that a gradient is a row vector.
 */
template <class T>
class Gradient
{
  typedef T BaseType;
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
    return std::string("\\nabla ") + t.latex(derivatives);
  }
};

template <class T>
Gradient<T>
grad(const T t)
{
  return Gradient<T>(t);
}

namespace Traits
{
  template <class T>
  struct is_test_function_set<Gradient<T>>
  {
    const static bool value = is_test_function_set<T>::value;
  };
}
}

#endif
