#include <cfl/dealii_matrixfree.h>

int
main()
{
  CFL::dealii::MatrixFree::FEFunction<0, 2, 0> fe_function_scalar("f_s");
  CFL::dealii::MatrixFree::FEFunction<1, 2, 0> fe_function_vector("f_v");
  CFL::dealii::MatrixFree::FEGradient<1, 2, 0> fe_gradient_scalar = grad(fe_function_scalar);
  CFL::dealii::MatrixFree::FEGradient<2, 2, 0> fe_gradient_vector = grad(fe_function_vector);
  CFL::dealii::MatrixFree::FEDivergence<0, 2, 0> fe_divergence = div(fe_function_vector);
  CFL::dealii::MatrixFree::FELiftDivergence<decltype(fe_divergence)> lifted_divergence(
    fe_divergence);
  CFL::dealii::MatrixFree::FESymmetricGradient<1, 2, 0> fe_symmetric_gradient("f_s_g_s");
  CFL::dealii::MatrixFree::FECurl<1, 2, 0> fe_curl("f_c_s");
  CFL::dealii::MatrixFree::FELaplacian<0, 2, 0> fe_laplacian = div(fe_gradient_scalar);
  CFL::dealii::MatrixFree::FEDiagonalHessian<2, 2, 0> fe_diagonal_hessian("fe_d_h_s");
  CFL::dealii::MatrixFree::FEHessian<2, 2, 0> fe_hessian("f_h_s");

  std::cout << fe_function_scalar.name() << std::endl;
  std::cout << fe_function_vector.name() << std::endl;
  std::cout << fe_gradient_scalar.name() << std::endl;
  std::cout << fe_gradient_vector.name() << std::endl;
  std::cout << fe_divergence.name() << std::endl;
  //  std::cout << lifted_divergence.name() << std::endl;
  std::cout << fe_symmetric_gradient.name() << std::endl;
  std::cout << fe_curl.name() << std::endl;
  std::cout << fe_laplacian.name() << std::endl;
  std::cout << fe_diagonal_hessian.name() << std::endl;
  std::cout << fe_hessian.name() << std::endl;
}
