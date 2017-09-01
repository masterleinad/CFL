#ifndef LATEX_EVALUATOR_H
#define LATEX_EVALUATOR_H

#include <vector>
#include <string>

#include <ostream>

namespace CFL::Latex
{

template<class FormContainer>
class Evaluator
{
public:
  Evaluator (const FormContainer& form_container,
                  const std::vector<std::string> & function_names,
                  const std::vector<std::string> & test_names)
    : _form_container(form_container),
      _function_names(function_names),
      _test_names(test_names)
  {}

  void print(std::ostream& out)
  {
    out << _form_container.print(_function_names, _test_names) << std::endl;
  }

private:
  const FormContainer& _form_container;
  const std::vector<std::string>& _function_names;
  const std::vector<std::string>& _test_names;
};
}

#endif // LATEX_EVALUATOR_H
