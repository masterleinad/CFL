#ifndef cfl_derivatives_h
#define cfl_derivatives_h

#include <array>
#include <cfl/traits.h>
#include <string>

namespace CFL
{
template <class T>
class Gradient;

namespace Traits
{
  template <class T>
  struct is_test_function_set<Gradient<T>>
  {
    const static bool value = is_test_function_set<T>::value;
  };

  template <class T>
  struct has_simple_derivative<Gradient<T>>
  {
    static const bool value = true;
  };

  template <class T>
  struct is_unary_operator<Gradient<T>>
  {
    static const bool value = true;
  };

  template <class T>
  struct is_cfl_object<Gradient<T>>
  {
    static const bool value = true;
  };

  template <class T>
  struct is_terminal_string<Gradient<T>>
  {
    static const bool value = is_terminal_string<T>::value;
  };
} // namespace Traits
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
public:
  using BaseType = T;
  const T base;

  typedef Traits::Tensor<T::TensorTraits::rank + 1, T::TensorTraits::dim> TensorTraits;

  explicit Gradient(const T& t)
    : base(t)
  {
    static_assert(Traits::has_simple_derivative<T>::value,
                  "Gradient can only take derivatives if those are simple");
  }

  template <typename... Comp>
  std::string
  latex(std::array<int, TensorTraits::dim> derivatives, int d, Comp... comp) const
  {
    // Discuss whether this copy is necessary, or if we want to hand
    // over derivatives by reference and screw it up for the caller
    ++derivatives[d];
    return base.latex(derivatives, comp...);
  }

  template <typename... Comp>
  std::string
  latex(int d, Comp... comp) const
  {
    std::array<int, TensorTraits::dim> derivatives({ { 0 } });
    derivatives[d] = 1;
    return base.latex(derivatives, comp...);
  }
};

template <class T>
Gradient<T>
grad(const T t)
{
  return Gradient<T>(t);
}
} // namespace CFL

#endif
