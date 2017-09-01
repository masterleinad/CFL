#ifndef cfl_products_h
#define cfl_products_h

#include <cfl/traits.h>

namespace CFL
{
/**
 * Multiplication of a scalar with any tensor.
 *
 * Since in this case the order of multiplication is irrelevant, we
 * always keep the scalar in the front.
 */
template <class S, class T>
class ScalarMultiplication
{
public:
  S a;
  T b;

  using TensorTraits = typename T::TensorTraits;

  ScalarMultiplication(const S& s, const T& t)
    : a(s)
    , b(t)
  {
    static_assert(S::TensorTraits::rank == 0,
                  "First argument must be a scalar with trensor rank zero");
  }

  template <typename... Comp>
  std::string
  latex(Comp... comp) const
  {
    return a.latex() + " " + b.latex(comp...);
  }
};

template <class S, class T>
typename std::enable_if<S::TensorTraits::rank == 0, ScalarMultiplication<S, T>>::type
multiply(const S& s, const T& t)
{
  return ScalarMultiplication<S, T>(s, t);
}

template <class S, class T>
typename std::enable_if<S::TensorTraits::rank == 0 && T::TensorTraits::rank != 0,
                        ScalarMultiplication<S, T>>::type
multiply(const T& t, const S& s)
{
  return ScalarMultiplication<S, T>(s, t);
}

namespace Traits
{
  template <class S, class T>
  struct is_binary_operator<ScalarMultiplication<S, T>>
  {
    static const bool value = true;
  };
} // namespace Traits
} // namespace CFL

#endif
