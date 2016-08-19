#ifndef cfl_traits_h
#define cfl_traits_h

namespace CFL
{
namespace Traits
{
template <int rank, int dim>
struct Tensor
{
  const unsigned int tensor_rank = rank;
  const unsigned int tensor_dim = dim;
};
}
}

#endif
