// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#define DEBUG_OUTPUT

#include <cfl/fefunctions.h>
#include <cfl/forms.h>

//#include <fstream>

#include <latex/evaluator.h>
#include <latex/fefunctions.h>
#include <latex/forms.h>

using namespace CFL;

void
test()
{
  constexpr Base::TestFunction<0, 1, 0> v;
  constexpr auto Dv = grad(v);
  constexpr Base::TestFunctionInteriorFace<0, 1, 0> v_p;
  constexpr Base::TestFunctionExteriorFace<0, 1, 0> v_m;
  constexpr Base::TestNormalGradientInteriorFace<0, 1, 0> Dnv_p;
  constexpr Base::TestNormalGradientExteriorFace<0, 1, 0> Dnv_m;

  constexpr Base::FEFunction<0, 1, 0> u;
  constexpr auto Du = grad(u);
  constexpr Base::FEFunctionInteriorFace<0, 1, 0> u_p;
  constexpr Base::FEFunctionExteriorFace<0, 1, 0> u_m;
  constexpr Base::FENormalGradientInteriorFace<0, 1, 0> Dnu_p;
  constexpr Base::FENormalGradientExteriorFace<0, 1, 0> Dnu_m;

  constexpr auto cell = Base::form(Du, Dv);

  constexpr auto flux = u_p - u_m;
  constexpr auto flux_grad = Dnu_p - Dnu_m;

  constexpr auto flux1 = -face_form(flux, Dnv_p) + Base::face_form(flux, Dnv_m);
  constexpr auto flux2 =
    Base::face_form(-flux + .5 * flux_grad, v_p) - Base::face_form(-flux + .5 * flux_grad, v_m);

  constexpr auto boundary1 = Base::boundary_form(2. * u_p - Dnu_p, v_p);
  constexpr auto boundary3 = -boundary_form(u_p, Dnv_p);

  constexpr auto face = -flux2 + .5 * flux1;
  constexpr auto f = cell + face + boundary1 + boundary3;

  std::vector<std::string> function_names(1, "u");
  std::vector<std::string> test_names(1, "v");
  constexpr auto latex_forms = Latex::transform(f);
  Latex::Evaluator<decltype(latex_forms)> evaluator(latex_forms, function_names, test_names);
  evaluator.print(std::cout);
}

int
main(int /*argc*/, char** /*argv*/)
{
  dealii::deallog.depth_console(10);
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
