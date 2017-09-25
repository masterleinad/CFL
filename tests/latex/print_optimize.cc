// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#define DEBUG_OUTPUT

#include <cfl/base/fefunctions.h>
#include <cfl/base/forms.h>

//#include <fstream>

#include <cfl/latex/evaluator.h>
#include <cfl/latex/fefunctions.h>
#include <cfl/latex/forms.h>

using namespace CFL;

void
test()
{
  constexpr unsigned int dim=3;
  constexpr Base::TestFunction<1, dim, 1> v;
  constexpr Base::TestFunction<0, dim, 0> q;
  
  constexpr Base::FEFunction<1, dim, 1> u;
  constexpr Base::FEFunction<0, dim, 0> p;
  constexpr auto Du = grad(u);
  constexpr auto divu = div(u);
  constexpr Base::FELiftDivergence<decltype(p)> p_lifted(p);

  constexpr auto sum1 = optimize(p+p);
  constexpr auto sum2 = optimize(p*p+divu+p*p);
  constexpr auto sum3 = optimize((p+divu+p)*p);

  constexpr auto total_sum = sum2/*+sum2+sum3*/;

  constexpr auto cell = Base::form(total_sum, q);

  std::vector<std::string> function_names{"p", "u"};
  std::vector<std::string> test_names{"q", "v"};
  constexpr auto latex_forms = Latex::transform(cell);
  Latex::Evaluator<decltype(latex_forms)> evaluator(latex_forms, function_names, test_names);
  evaluator.print(std::cout);
}

int
main(int /*argc*/, char** /*argv*/)
{
  try
  {
    test();
  }
  catch (std::exception& exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
