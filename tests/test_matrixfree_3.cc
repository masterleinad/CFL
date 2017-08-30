///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_3_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case FEObjMemFunc
// Type: Positive test case
// Coverage: following classes - FEFunction, FEDivergence, FELiftDivergence,
// 				FESymmetricGradient,FECurl, FEGradient, FELaplacian,
// FEDiagonalHessian,
//             FEHessain, FEFunctionInteriorFace, FEFunctionExteriorFace,
//				FENormalGradientInteriorFace, FENormalGradientExteriorFace
// 				for operator- and operator *
// Not tested: functions value() and set_evaluation_flags() since as a prerequisite,
// we will need to create a MatrixFree object, and a complete solver
BOOST_FIXTURE_TEST_CASE(FEObjMemFunc, FEFixture, *utf::tolerance(0.00001))
{
  // This is to invoke template hierarchy of FEDatas
  // which are stored in FEFunction object
  auto fedatas1 = (fedatas, fedata_1_system);

  FEFunction<0, dim, fe_no_0> fe_function_scalar1("f_s");
  auto fe_function_scalar2 = -fe_function_scalar1;
  BOOST_TEST(fe_function_scalar2.scalar_factor == -fe_function_scalar1.scalar_factor);
  auto fe_function_scalar3 = 0.2 * fe_function_scalar1 * 0.1;
  BOOST_TEST(fe_function_scalar3.scalar_factor == fe_function_scalar1.scalar_factor * 0.1 * 0.2);

  FEFunction<1, dim, fe_no_0> fe_function_vector1("f_v");
  auto fe_function_vector2 = -fe_function_vector1;
  BOOST_TEST(fe_function_vector2.scalar_factor == -fe_function_vector1.scalar_factor);
  auto fe_function_vector3 = 0.2 * fe_function_vector1 * 0.1;
  BOOST_TEST(fe_function_vector3.scalar_factor == fe_function_vector1.scalar_factor * 0.1 * 0.2);

  FEGradient<1, dim, fe_no_0> fe_gradient_scalar1 = grad(fe_function_scalar1);
  auto fe_gradient_scalar2 = -fe_gradient_scalar1;
  BOOST_TEST(fe_gradient_scalar2.scalar_factor == -fe_gradient_scalar1.scalar_factor);
  auto fe_gradient_scalar3 = 0.2 * fe_gradient_scalar1 * 0.1;
  BOOST_TEST(fe_gradient_scalar3.scalar_factor == fe_gradient_scalar1.scalar_factor * 0.1 * 0.2);

  FEGradient<2, dim, fe_no_0> fe_gradient_vector1 = grad(fe_function_vector1);
  auto fe_gradient_vector2 = -fe_gradient_vector1;
  BOOST_TEST(fe_gradient_vector2.scalar_factor == -fe_gradient_vector1.scalar_factor);
  auto fe_gradient_vector3 = 0.2 * fe_gradient_vector1 * 0.1;
  BOOST_TEST(fe_gradient_vector3.scalar_factor == fe_gradient_vector1.scalar_factor * 0.1 * 0.2);

  FEDivergence<0, dim, fe_no_0> fe_divergence1 = div(fe_function_vector1);
  auto fe_divergence2 = -fe_divergence1;
  BOOST_TEST(fe_divergence2.scalar_factor == -fe_divergence1.scalar_factor);
  auto fe_divergence3 = 0.2 * fe_divergence1 * 0.1;
  BOOST_TEST(fe_divergence3.scalar_factor == fe_divergence1.scalar_factor * 0.1 * 0.2);

  FELiftDivergence<decltype(fe_divergence1)> lifted_divergence1(fe_divergence1);
  auto lifted_divergence2 = -lifted_divergence1;
  // Sorry, we cant check the outcome like below in UT since this class has different interface than
  // rest
  // BOOST_TEST(lifted_divergence2.scalar_factor == -lifted_divergence1.scalar_factor);
  auto lifted_divergence3 = 0.2 * lifted_divergence1 * 0.1;
  // Sorry, we cant check the outcome like below in UT since this class has different interface than
  // rest
  // BOOST_TEST(lifted_divergence3.scalar_factor == lifted_divergence1.scalar_factor * 0.1 * 0.2);

  FESymmetricGradient<1, dim, fe_no_0> fe_symmetric_gradient1("f_s_g_s");
  auto fe_symmetric_gradient2 = -fe_symmetric_gradient1;
  BOOST_TEST(fe_symmetric_gradient2.scalar_factor == -fe_symmetric_gradient1.scalar_factor);
  auto fe_symmetric_gradient3 = 0.2 * fe_symmetric_gradient1 * 0.1;
  BOOST_TEST(fe_symmetric_gradient3.scalar_factor ==
             fe_symmetric_gradient1.scalar_factor * 0.1 * 0.2);

  FECurl<1, dim, fe_no_0> fe_curl1("f_c_s");
  auto fe_curl2 = -fe_curl1;
  BOOST_TEST(fe_curl2.scalar_factor == -fe_curl1.scalar_factor);
  auto fe_curl3 = 0.2 * fe_curl1 * 0.1;
  BOOST_TEST(fe_curl3.scalar_factor == fe_curl1.scalar_factor * 0.1 * 0.2);

  FELaplacian<0, dim, fe_no_0> fe_laplacian1 = div(fe_gradient_scalar1);
  auto fe_laplacian2 = -fe_laplacian1;
  BOOST_TEST(fe_laplacian2.scalar_factor == -fe_laplacian1.scalar_factor);
  auto fe_laplacian3 = 0.2 * fe_laplacian1 * 0.1;
  BOOST_TEST(fe_laplacian3.scalar_factor == fe_laplacian1.scalar_factor * 0.1 * 0.2);

  FEDiagonalHessian<2, dim, fe_no_0> fe_diagonal_hessian1("fe_d_h_s");
  auto fe_diagonal_hessian2 = -fe_diagonal_hessian1;
  BOOST_TEST(fe_diagonal_hessian2.scalar_factor == -fe_diagonal_hessian1.scalar_factor);
  auto fe_diagonal_hessian3 = 0.2 * fe_diagonal_hessian1 * 0.1;
  BOOST_TEST(fe_diagonal_hessian3.scalar_factor == fe_diagonal_hessian1.scalar_factor * 0.1 * 0.2);

  FEHessian<2, dim, fe_no_0> fe_hessian1("f_h_s");
  auto fe_hessian2 = -fe_hessian1;
  BOOST_TEST(fe_hessian2.scalar_factor == -fe_hessian1.scalar_factor);
  auto fe_hessian3 = 0.2 * fe_hessian1 * 0.1;
  BOOST_TEST(fe_hessian3.scalar_factor == fe_hessian1.scalar_factor * 0.1 * 0.2);

  FEFunctionInteriorFace<2, dim, fe_no_0> fe_int_face1("f_in_f");
  auto fe_int_face2 = -fe_int_face1;
  BOOST_TEST(fe_int_face2.scalar_factor == -fe_int_face1.scalar_factor);
  auto fe_int_face3 = 0.2 * fe_int_face1 * 0.1;
  BOOST_TEST(fe_int_face3.scalar_factor == fe_int_face1.scalar_factor * 0.1 * 0.2);

  FEFunctionExteriorFace<2, dim, fe_no_0> fe_ext_face1("f_ex_f");
  auto fe_ext_face2 = -fe_ext_face1;
  BOOST_TEST(fe_ext_face2.scalar_factor == -fe_ext_face1.scalar_factor);
  auto fe_ext_face3 = 0.2 * fe_ext_face1 * 0.1;
  BOOST_TEST(fe_ext_face3.scalar_factor == fe_ext_face1.scalar_factor * 0.1 * 0.2);

  FENormalGradientInteriorFace<2, dim, fe_no_0> fe_gint_face1("f_g_in_f");
  auto fe_gint_face2 = -fe_gint_face1;
  BOOST_TEST(fe_gint_face2.scalar_factor == -fe_gint_face1.scalar_factor);
  auto fe_gint_face3 = 0.2 * fe_gint_face1 * 0.1;
  BOOST_TEST(fe_gint_face3.scalar_factor == fe_gint_face1.scalar_factor * 0.1 * 0.2);

  FENormalGradientExteriorFace<2, dim, fe_no_0> fe_gext_face1("f_g_in_f");
  auto fe_gext_face2 = -fe_gint_face1;
  BOOST_TEST(fe_gext_face2.scalar_factor == -fe_gext_face1.scalar_factor);
  auto fe_gext_face3 = 0.2 * fe_gext_face1 * 0.1;
  BOOST_TEST(fe_gext_face3.scalar_factor == fe_gext_face1.scalar_factor * 0.1 * 0.2);
}
