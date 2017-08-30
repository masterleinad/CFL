#ifndef cfl_transpose_h
#define cfl_transpose_h

namespace CFL
{
template <class T>
class Transpose
{
  const T& t;
  const unsigned int i1;
  const unsigned int i2;

  typedef T::Traits Traits;
  Traits traits;

public:
  Transpose(const T& t, const unsigned int index1 = 0, const unsigned int index2 = 1)
    : t(t)
    , i1(index1)
    , i2(index2)
  {
  }

  std::string
  latex(std::array<int, dim> derivatives, Components... comp) const
  {
  }
};
}

#endif
