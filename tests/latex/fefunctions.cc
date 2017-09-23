// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include <cfl/base/fefunctions.h>
#include <cfl/latex/fefunctions.h>

using namespace CFL;

int
main()
{
  Base::FEFunction<0, 2, 0> fe_function_scalar;
  Base::FEFunction<1, 2, 1> fe_function_vector;
  Base::FEGradient<1, 2, 0> fe_gradient_scalar = grad(fe_function_scalar);
  Base::FEGradient<2, 2, 1> fe_gradient_vector = grad(fe_function_vector);
  Base::FEDivergence<0, 2, 1> fe_divergence = div(fe_function_vector);
  Base::FELiftDivergence<decltype(fe_divergence)> lifted_divergence(fe_divergence);
  Base::FESymmetricGradient<1, 2, 0> fe_symmetric_gradient;
  Base::FECurl<1, 2, 0> fe_curl;
  Base::FELaplacian<0, 2, 0> fe_laplacian = div(fe_gradient_scalar);
  Base::FEDiagonalHessian<2, 2, 0> fe_diagonal_hessian;
  Base::FEHessian<2, 2, 0> fe_hessian;

  std::vector<std::string> function_names{ "s", "v" };

  static_assert(CFL::Traits::is_cfl_object<decltype(fe_function_scalar)>::value);

  std::cout << Latex::transform(fe_function_scalar).value(function_names) << std::endl;
  std::cout << Latex::transform(fe_function_vector).value(function_names) << std::endl;
  std::cout << Latex::transform(fe_gradient_scalar).value(function_names) << std::endl;
  std::cout << Latex::transform(fe_gradient_vector).value(function_names) << std::endl;
  std::cout << Latex::transform(fe_divergence).value(function_names) << std::endl;
  std::cout << Latex::transform(lifted_divergence).value(function_names) << std::endl;
  std::cout << Latex::transform(fe_symmetric_gradient).value(function_names) << std::endl;
  std::cout << Latex::transform(fe_curl).value(function_names) << std::endl;
  std::cout << Latex::transform(fe_laplacian).value(function_names) << std::endl;
  std::cout << Latex::transform(fe_diagonal_hessian).value(function_names) << std::endl;
  std::cout << Latex::transform(fe_hessian).value(function_names) << std::endl;
}
