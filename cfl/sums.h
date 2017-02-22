#ifndef cfl_sums_h
#define cfl_sums_h

#include <array>
#include <cfl/traits.h>
#include <string>

namespace CFL
{
template <class A, class B>
class Sum
{
  const A& a;
  const B& b;

public:
  typedef typename Traits::Tensor<A::TensorTraits::rank, A::TensorTraits::dim> TensorTraits;

  Sum(const A& a, const B& b)
    : a(a)
    , b(b)
  {
    static_assert(!::CFL::Traits::is_test_function_set<A>::value, "Test functions cannot be added");
    static_assert(!::CFL::Traits::is_test_function_set<B>::value, "Test functions cannot be added");

    static_assert(A::TensorTraits::rank == B::TensorTraits::rank,
                  "You can only add tensors of equal rank");
    static_assert(A::TensorTraits::dim == B::TensorTraits::dim,
                  "You can only add tensors of equal dimension");
  }

  template <typename... Comp>
  std::string
  latex(Comp... comp) const
  {
    std::string output = a.latex(comp...) + "+" + b.latex(comp...);
    return output;
  }

  template <class FEEvaluation>
  auto value(const FEEvaluation& phi, unsigned int q) const;
};

template <class A, class B>
template <class FEEvaluation>
auto
Sum<A, B>::value(const FEEvaluation& phi, unsigned int q) const
{
  return a.value(phi, q) + b.value(phi, q);
}

namespace Traits
{
  template <class T, class U>
  struct is_terminal_string<Sum<T, U>>
  {
    static const bool value = is_terminal_string<T>::value && is_terminal_string<U>::value;
  };
} // namespace Traits

template <class A, class B>
typename std::enable_if_t<((CFL::Traits::is_cfl_object<A>::value ||
                            CFL::Traits::is_cfl_object<B>::value) &&
                           !CFL::Traits::is_summable<A, B>::value),
                          A>
operator+(const A& a, const B& b)
{
  assert_is_summable(a, b);
  throw std::runtime_error("Internal error! This line should never be invoked!");
  return a;
}
} // namespace CFL

#endif
