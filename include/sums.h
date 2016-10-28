#ifndef cfl_sums_h
#define cfl_sums_h

#include <array>
#include <string>
#include <traits.h>

namespace CFL
{

template <class A, class B>
class Sum
{
  const A a;
  const B b;

public:
  typedef Traits::Tensor<A::Traits::rank, A::Traits::dim> Traits;
  Traits traits;

  Sum(const A& a, const B& b)
    : a(a)
    , b(b)
  {
    static_assert(!CFL::Traits::is_test_function_set<A>::value, "Test functions cannot be added");
    static_assert(!CFL::Traits::is_test_function_set<B>::value, "Test functions cannot be added");

    static_assert(A::Traits::rank == B::Traits::rank, "You can only add tensors of equal rank");
    static_assert(A::Traits::dim == B::Traits::dim, "You can only add tensors of equal dimension");
  }

  template <typename... Comp>
  std::string latex(Comp... comp) const
  {
    std::string output = a.latex(comp...) + "+" + b.latex(comp...);
    return output;
  }
};

template <class A, class B>
Sum<A, B>
operator+(const A& a, const B& b)
{
  return Sum<A, B>(a, b);
}
}

#endif
