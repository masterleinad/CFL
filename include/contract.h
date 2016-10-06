#ifndef cfl_contract_h
#define cfl_contract_h

namespace CFL
{
template <class A, class B>
class Contract
{
  const A& a;
  const B& b;
  const unsigned int level;
public:
  Contract(const A&a, const B&b, unsigned int level)
    : a(a), b(b), level(level)
  {}

};
}

#endif
