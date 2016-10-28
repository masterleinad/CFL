#ifndef cfl_terminal_strings_h
#define cfl_terminal_strings_h

#include <array>
#include <string>

#include <traits.h>

namespace CFL
{
template <typename... Components>
std::string compose_indices(Components...)
{
  return std::string();
}

template <typename... Components>
std::string
compose_indices(int last, Components... comp)
{
  return compose_indices(comp...) + std::to_string(last);
}

/**
 * A terminal object for expression templates producing LaTeX
 * output.
 *
 * These objects have their tensor rank and dimension, such that
 * expressions of the form $u_{i_1 i_2 \dots i_r}$ make sense, where
 * $r$ is the rank and each index $i_k$ is between 0 and
 * <tt>dim</tt>.
 *
 * The template argument <tt>is_test</tt> makes the object a test
 * function set in order to output forms etc. correctly.
 */
template <int rank, int dim, bool is_test = false>
class TerminalString
{
  std::string value;

public:
  typedef Traits::Tensor<rank, dim> Traits;
  Traits traits;

  TerminalString(const std::string val)
    : value(val)
  {
  }

  template <typename... Components>
  std::string
  latex(std::array<int, dim> derivatives, Components... comp) const
  {
    std::string result;
    for (unsigned int d = 0; d < dim; ++d)
      if (derivatives[d] != 0)
      {
        result += std::string("\\partial_{") + std::to_string(d) + std::string("}");
        if (derivatives[d] > 1)
          result += std::string("^{") + std::to_string(derivatives[d]) + std::string("}");
      }
    result += value;
    if (rank > 0)
    {
      result += std::string("_{") + compose_indices(comp...) + std::string("}");
    }
    return result;
  }
};

namespace Traits
{
  template <int rank, int dim>
  struct is_test_function_set<TerminalString<rank, dim, true>>
  {
    const static bool value = true;
  };
}
}

#endif
