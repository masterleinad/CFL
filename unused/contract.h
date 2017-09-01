#ifndef cfl_contract_h
#define cfl_contract_h

namespace CFL
{
/**
 * Allows to contract input tensors <code>A</code> and <code>B</code>
 * over <code>level</code> dimensions
 *
 * @note Currently unused
 *
 * @todo Seems to be incomplete..discuss
 * 1. Expected result of contraction?
 * 2. Constraints on inputs?
 *
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
