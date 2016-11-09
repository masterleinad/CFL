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
    const A a;
    const B b;

  public:
    typedef Traits::Tensor<A::TensorTraits::rank, A::TensorTraits::dim> TensorTraits;

    Sum(const A& a, const B& b)
      : a(a)
      , b(b)
    {
      static_assert(!::CFL::Traits::is_test_function_set<A>::value, "Test functions cannot be added");
      static_assert(!::CFL::Traits::is_test_function_set<B>::value, "Test functions cannot be added");

      static_assert(A::TensorTraits::rank == B::TensorTraits::rank, "You can only add tensors of equal rank");
      static_assert(A::TensorTraits::dim == B::TensorTraits::dim, "You can only add tensors of equal dimension");
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
