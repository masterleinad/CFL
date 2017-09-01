#ifndef cfl_products_h
#define cfl_products_h

#include <cfl/traits.h>

#include <stdexcept>

namespace CFL
{
template <class A, class B>
typename std::enable_if_t<((CFL::Traits::is_cfl_object<A>::value ||
                            CFL::Traits::is_cfl_object<B>::value) &&
                           !CFL::Traits::is_multiplicable<A, B>::value),
                          A>
operator*(const A& a, const B& b)
{
  assert_is_multiplicable(a, b);
  throw std::runtime_error("Internal error! This line should never be invoked!");
  return a;
}

} // namespace CFL

#endif
