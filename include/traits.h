#ifndef cfl_traits_h
#define cfl_traits_h

namespace CFL
{
namespace Traits
{
struct Scalar
{
  const unsigned int tensor_rank = 0;
  constexpr unsigned int
  tensor_dimension(unsigned int)
  {
    return 1;
  }
};

template <int rank, int dim>
struct Tensor
{
  const unsigned int tensor_rank = rank;
  constexpr unsigned int
  tensor_dimension(unsigned int)
  {
    return dim;
  }
};
}
}

#endif
