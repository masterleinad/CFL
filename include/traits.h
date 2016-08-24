#ifndef cfl_traits_h
#define cfl_traits_h

namespace CFL
{
namespace Traits
{
template <int rank_, int dim_>
struct Tensor
{
  static const unsigned int rank = rank_;
  static const unsigned int dim = dim_;
};
}
}

#endif
