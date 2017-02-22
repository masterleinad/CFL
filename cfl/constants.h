
#ifndef cfl_constants_h
#define cfl_constants_h

#include <cfl/traits.h>
#include <string>

namespace CFL
{
template <typename number, typename T>
class ConstantScaled
{
  const number factor;
  const T t;

public:
  using TensorTraits = typename T::TensorTraits;
  constexpr ConstantScaled(const number& factor, const T& t)
    : factor(factor)
    , t(t)
  {
  }

  template <typename... Comp>
  std::string
  latex(Comp... comp) const
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
} // namespace CFL

#endif
