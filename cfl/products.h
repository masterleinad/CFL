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
  S scalar;
  T other;

public:
  typedef typename T::Traits Traits;
  Traits traits;

  ScalarMultiplication(const S& s, const T& t)
    : scalar(s)
    , other(t)
  {
    static_assert(S::Traits::rank == 0, "First argument must be a scalar with trensor rank zero");
  }

  template <typename... Comp>
  std::string latex(Comp... comp) const
  {
    return scalar.latex() + " " + other.latex(comp...);
  }
};

template <class S, class T>
typename std::enable_if<S::Traits::rank == 0, ScalarMultiplication<S, T>>::type
multiply(const S& s, const T& t)
{
  return ScalarMultiplication<S, T>(s, t);
}

template <class S, class T>
typename std::enable_if<S::Traits::rank == 0 && T::Traits::rank != 0,
                        ScalarMultiplication<S, T>>::type
multiply(const T& t, const S& s)
{
  return ScalarMultiplication<S, T>(s, t);
}

template <class A, class B>
auto operator*(const A& a, const B& b)
{
  return multiply(a, b);
}
}

#endif
