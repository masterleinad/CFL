///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_3_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case FEObjMemFunc
// Type: Positive test case
// Coverage: following classes - FEFunction, FEDivergence, FELiftDivergence, FESymmetricGradient,FECurl
//             					 FEGradient, FELaplacian, FEDiagonalHessian, FEHessain
//			 following functions - div(), grad()
// Checks for:
// 1. Template instantiation of all member functions
//		Typically there is nothing more to test since the member functions ultimately call dealii library functions.
//		We only test state of our object and not dealii results
BOOST_FIXTURE_TEST_CASE(FEObjMemFunc,FEFixture,* utf::tolerance(0.00001)) {

	auto fedatas1 = (fedatas,fedata_1_system); //This is to invoke template hierarchy of FEDatas which are stored in FEFunction object

	FEFunction<0, dim, fe_no_0> fe_function_scalar("f_s");
	fe_function_scalar.value(fedatas,1);
	//fe_function_scalar.set_evaluation_flags(); possible Issue discuss..point 4, and add more cases for each class
	FEFunction<0, dim, fe_no_1> fe_function_scalar1("f_s");
	fe_function_scalar1.value(fedatas1,1);
	//TBD . behavior comments - see point 8
	auto fe_function_scalar2 = -fe_function_scalar;
	BOOST_TEST(fe_function_scalar2.scalar_factor == -fe_function_scalar.scalar_factor);
	auto fe_function_scalar3 = 0.2 * fe_function_scalar * 0.1;
	BOOST_TEST(fe_function_scalar3.scalar_factor == fe_function_scalar.scalar_factor*0.1*0.2);


	FEFunction<1, dim, fe_no_0> fe_function_vector("f_v");
	fe_function_vector.value(fedatas,1);
	FEFunction<1, dim, fe_no_1> fe_function_vector1("f_v");
	fe_function_vector1.value(fedatas1,1);
	auto fe_function_vector2 = -fe_function_vector;
	BOOST_TEST(fe_function_vector2.scalar_factor == -fe_function_vector.scalar_factor);
	auto fe_function_vector3 = 0.2 * fe_function_vector * 0.1;
	BOOST_TEST(fe_function_vector3.scalar_factor == fe_function_vector.scalar_factor*0.1*0.2);



	FEGradient<1, dim, fe_no_0> fe_gradient_scalar = grad(fe_function_scalar);
	fe_gradient_scalar.value(fedatas,1);
	FEGradient<1, dim, fe_no_1> fe_gradient_scalar1 = grad(fe_function_scalar1);
	fe_gradient_scalar1.value(fedatas1,1);
	auto fe_gradient_scalar2 = -fe_gradient_scalar;
	BOOST_TEST(fe_gradient_scalar2.scalar_factor == -fe_gradient_scalar.scalar_factor);
	auto fe_gradient_scalar3 = 0.2 * fe_gradient_scalar * 0.1;
	BOOST_TEST(fe_gradient_scalar3.scalar_factor == fe_gradient_scalar.scalar_factor*0.1*0.2);



	FEGradient<2, dim, fe_no_0> fe_gradient_vector = grad(fe_function_vector);
	fe_gradient_vector.value(fedatas,1);
	FEGradient<2, dim, fe_no_1> fe_gradient_vector1 = grad(fe_function_vector1);
	fe_gradient_vector1.value(fedatas1,1);
	auto fe_gradient_vector2 = -fe_gradient_vector;
	BOOST_TEST(fe_gradient_vector2.scalar_factor == -fe_gradient_vector.scalar_factor);
	auto fe_gradient_vector3 = 0.2 * fe_gradient_vector * 0.1;
	BOOST_TEST(fe_gradient_vector3.scalar_factor == fe_gradient_vector.scalar_factor*0.1*0.2);


	//TBD: fails in get_divergence missing definition, same as in fe_data.h
	//FEDivergence<0, dim, fe_no_0> fe_divergence = div(fe_function_vector);
	//fe_divergence.value(fedatas,1);

	//FELiftDivergence<decltype(fe_divergence)> lifted_divergence(fe_divergence); //TBD, yet to analyze

	//TBD: Fails in get_symmetric_gradient missing definition, same as in fe_data.h
	//FESymmetricGradient<1, dim, fe_no_0> fe_symmetric_gradient("f_s_g_s");
	//fe_symmetric_gradient.value(fedatas,1);

	//TBD: No definition of get_curl. Discuss, point no. 5
	//FECurl<1, dim, fe_no_0> fe_curl("f_c_s");
	//fe_curl.value(fedatas,1);

	FELaplacian<0, dim, fe_no_0> fe_laplacian = div(fe_gradient_scalar);
	fe_laplacian.value(fedatas,1);
	FELaplacian<0, dim, fe_no_1> fe_laplacian1 = div(fe_gradient_scalar1);
	fe_laplacian1.value(fedatas1,1);
	auto fe_laplacian2 = -fe_laplacian;
	BOOST_TEST(fe_laplacian2.scalar_factor == -fe_laplacian.scalar_factor);
	auto fe_laplacian3 = 0.2 * fe_laplacian * 0.1;
	BOOST_TEST(fe_laplacian3.scalar_factor == fe_laplacian.scalar_factor*0.1*0.2);


	FEDiagonalHessian<2, dim, fe_no_0> fe_diagonal_hessian("fe_d_h_s");
	fe_diagonal_hessian.value(fedatas,1);
	FEDiagonalHessian<2, dim, fe_no_1> fe_diagonal_hessian1("fe_d_h_s");
	fe_diagonal_hessian1.value(fedatas1,1);
	auto fe_diagonal_hessian2 = -fe_diagonal_hessian;
	BOOST_TEST(fe_diagonal_hessian2.scalar_factor == -fe_diagonal_hessian.scalar_factor);
	auto fe_diagonal_hessian3 = 0.2 * fe_diagonal_hessian * 0.1;
	BOOST_TEST(fe_diagonal_hessian3.scalar_factor == fe_diagonal_hessian.scalar_factor*0.1*0.2);


	FEHessian<2, dim, fe_no_0> fe_hessian("f_h_s");
	fe_hessian.value(fedatas,1);
	FEHessian<2, dim, fe_no_1> fe_hessian1("f_h_s");
	fe_hessian1.value(fedatas1,1);
	auto fe_hessian2 = -fe_hessian;
	BOOST_TEST(fe_hessian2.scalar_factor == -fe_hessian.scalar_factor);
	auto fe_hessian3 = 0.2 * fe_hessian * 0.1;
	BOOST_TEST(fe_hessian3.scalar_factor == fe_hessian.scalar_factor*0.1*0.2);
}

