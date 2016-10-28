#ifndef cfl_contract_h
#define cfl_contract_h

namespace CFL
{
/**
 * Contract tensors A and B over level dimensions.
 */
template <class A, class B, int level = A::Traits::rank>
class Contract
{
  const A& a;
  const B& b;

public:
  Contract(const A& a, const B& b)
    : a(a)
    , b(b)
  {
    static_assert(level <= A::Traits::rank, "Level cannot exceed rank");
    static_assert(level <= B::Traits::rank, "Level cannot exceed rank");
  }
};
}

#endif
