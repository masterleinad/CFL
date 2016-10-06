#ifndef cfl_inner_product_h
#define cfl_inner_product_h

namespace CFL
{
template <class A, class B>
class InnerProduct
{
  const A& a;
  const B& b;
public:
  InnerProduct(const A&a, const B&b)
    : a(a), b(b)
  {}
  
};
}

#endif
