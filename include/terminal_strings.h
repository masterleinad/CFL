#ifndef cfl_terminal_strings_h
#define cfl_terminal_strings_h

#include <string>
#include <array>

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

template <int rank, int dim>
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
}

#endif
