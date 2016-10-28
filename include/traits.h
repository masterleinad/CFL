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

  template <class T>
  struct is_test_function_set
  {
    static const bool value = false;
  };

  template <class T>
  struct is_form
  {
    static const bool value = false;
  };
}
}

#endif
