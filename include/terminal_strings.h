#ifndef cfl_terminal_strings_h
#define cfl_terminal_strings_h

#include<string>
#include<array>

#include <traits.h>

namespace CFL
{
template <int rank, int dim>
class TerminalString
{
  std::string value;
 public:

  typedef Traits::Tensor<rank,dim> traits;
  
  TerminalString(const std::string val)
    : value(val)
  {}
  
  std::string latex(std::array<int,rank> components,
		    std::array<int,dim> derivatives) const
  {
    std::string result;
    for (unsigned int d=0;d<dim;++d)
      if (derivatives[d]!=0)
	{
	  result += std::string("\\partial_{")
	    + std::to_string(d) + std::string("}");
	  if (derivatives[d]>1)
	    result += std::string("^{") + std::to_string(derivatives[d])
	      + std::string("}");
	}
    result += value;
    if (rank>0)
      {
	result += std::string("_{");
	for (unsigned int r=0;r<rank;++r)
	  result += std::to_string(components[r]);
	    result += std::string("}");
      }
    return result;
  }
};
}

#endif
