
#ifndef cfl_constants_h
#define cfl_constants_h

#include <cfl/traits.h>
#include <string>

namespace CFL
{

/**
 * Allows to store an object of user defined type <code> T </code> and a
 * scaling factor to be applied on it. The exact application of scaling factor
 * is left to user implementation
 *
 * @note Currently unused
 *
 * @todo "Any" type is allowed as scaling factor. It would be good to validate
 * and restrict to a subset of numeric types
 *
 */
  template <typename T, typename number=double>
class ConstantScaled
{
  const number factor;
  const T t;

public:
  using TensorTraits = typename T::TensorTraits;
  constexpr ConstantScaled(const number& factor_, const T& t_)
    : factor(factor_)
    , t(t_)
  {
  }

  template <typename... Comp>
  std::string
  latex(Comp... comp) const
  {
    return std::to_string(factor) + " " + t.latex(comp...);
  }

  template <typename... Types>
  auto value (Types... args) const
  {
    return factor * t.value(args...);
  }
};

/**
 * \relates ConstantScaled
 * Utility function to scale an object of type <code>T</code> with a constant
 * <code>factor</code>.
 * Returns a new object of type \ref ConstantScaled.
 * The scaling is not actually performed but factor is retained as part of
 * \ref ConstantScaled for later (lazy) evaluation
 */
template <typename number, class T>
auto
scale(number factor, const T& t)
{
  return ConstantScaled<T, number>(factor, t);
}

  namespace Traits
  {
    template <typename T, typename number>
    struct fe_function_set_type<ConstantScaled<T, number>>
    {
      static const CFL::ObjectType value = fe_function_set_type<T>::value;
    };
  }
} // namespace CFL

#endif
