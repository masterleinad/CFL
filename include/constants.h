
#ifndef cfl_constants_h
#define cfl_constants_h

#include <string>
#include <traits.h>

namespace CFL
{
template <typename number, typename T>
class ConstantScaled
{
  const number factor;
  const T t;

public:
  typedef typename T::Traits Traits;
  Traits traits;
  constexpr ConstantScaled(const number& factor, const T& t)
    : factor(factor)
    , t(t)
  {
  }

  template <typename... Comp>
  std::string latex(Comp... comp) const
  {
    return std::to_string(factor) + " " + t.latex(comp...);
  }
};

template <typename number, class T>
auto
scale(number factor, const T& t)
{
  return ConstantScaled<number, T>(factor, t);
}
}

#endif
