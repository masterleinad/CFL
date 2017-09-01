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
  Base::TestFunction<0, 1, 0> v;
  auto Dv = grad(v);
  Base::TestFunctionInteriorFace<0, 1, 0> v_p;
  Base::TestFunctionExteriorFace<0, 1, 0> v_m;
  Base::TestNormalGradientInteriorFace<0, 1, 0> Dnv_p;
  Base::TestNormalGradientExteriorFace<0, 1, 0> Dnv_m;

  Base::FEFunction<0, 1, 0> u;
  auto Du = grad(u);
  Base::FEFunctionInteriorFace<0, 1, 0> u_p;
  Base::FEFunctionExteriorFace<0, 1, 0> u_m;
  Base::FENormalGradientInteriorFace<0, 1, 0> Dnu_p;
  Base::FENormalGradientExteriorFace<0, 1, 0> Dnu_m;

  auto cell = form(Du, Dv);

  auto flux = u_p - u_m;
  auto flux_grad = Dnu_p - Dnu_m;

  auto flux1 = -face_form(flux, Dnv_p) + face_form(flux, Dnv_m);
  auto flux2 = face_form(-flux + .5 * flux_grad, v_p) - face_form(-flux + .5 * flux_grad, v_m);

  auto boundary1 = boundary_form(2. * u_p - Dnu_p, v_p);
  auto boundary3 = -boundary_form(u_p, Dnv_p);

  auto face = -flux2 + .5 * flux1;
  auto f = cell + face + boundary1 + boundary3;

  std::vector<std::string> function_names(1, "u");
  std::vector<std::string> test_names(1, "v");
  const auto latex_forms = Latex::transform(f);
  Latex::Evaluator<decltype(latex_forms)> evaluator(latex_forms, function_names, test_names);
  evaluator.print(std::cout);
}

int
main(int /*argc*/, char** /*argv*/)
{
  ::dealii::deallog.depth_console(10);
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
