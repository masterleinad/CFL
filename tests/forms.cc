// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include <iostream>
#include <latex/evaluator.h>
#include <latex/fefunctions.h>
#include <latex/forms.h>

using namespace CFL;

int
main()
{
  Base::FEFunction<0, 2, 0> p("p");
  Base::TestFunction<0, 2, 0> q;
  Base::FEFunction<1, 2, 1> u("u");
  Base::TestFunction<1, 2, 1> v;

  std::vector<std::string> function_names{ "p", "u" };
  std::vector<std::string> test_names{ "q", "v" };

  Latex::Evaluator(Latex::transform(form(q, p)), function_names, test_names).print(std::cout);
  Latex::Evaluator(Latex::transform(form(p, q)), function_names, test_names).print(std::cout);
  Latex::Evaluator(Latex::transform(form(v, u)), function_names, test_names).print(std::cout);
  Latex::Evaluator(Latex::transform(form(grad(q), u)), function_names, test_names).print(std::cout);
  Latex::Evaluator(Latex::transform(form(v, grad(p))), function_names, test_names).print(std::cout);
  Latex::Evaluator(Latex::transform(form(grad(v), grad(u))), function_names, test_names)
    .print(std::cout);
  Latex::Evaluator(Latex::transform(form(grad(grad(q)), grad(u))), function_names, test_names)
    .print(std::cout);
  Latex::Evaluator(Latex::transform(form(grad(v), grad(grad(p)))), function_names, test_names)
    .print(std::cout);
  Latex::Evaluator(Latex::transform(form(grad(grad(q)), grad(grad(p)))), function_names, test_names)
    .print(std::cout);

  return 0;
}
