///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_5_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case ProdFEObjSameType
// Type: Positive test case
// Coverage: following classes - ProductFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "*" when types of operands is exactly same
//		This is done by "invoking" operator overloading for several combinations which can conclude that:
// 		- all possible ways to multiply FE functions is tested
//		- result of ProductFEFunction is of correct type
//		There are no check on values since they are available from dealii library function results which we ignore in out test
template <template <int,int,unsigned int> typename FEFuncType,int rank, int dim, unsigned int idx>
void prodfe_same_types() {
	using FEFunc = FEFuncType<rank, dim, idx>;

	FEFunc fe_function1("test_fe1");
	FEFunc fe_function2("test_fe2");
	FEFunc fe_function3("test_fe3");

	auto prod1 = fe_function1 * fe_function2;
	auto prod2 = fe_function2 * fe_function1;
	auto prod3 = fe_function2 * fe_function1 * fe_function3;
	auto prod4 = prod1 * fe_function1 * fe_function2;
#if 0 //See 15
	auto prod5 = fe_function1 * fe_function2 * prod1;
	auto prod6 = prod1 * prod2;
	auto prod7 = prod1 * prod2 * fe_function1;
	auto prod8 = fe_function2 * prod1 * prod2;
	auto prod9 = prod6 * prod8;
#endif
}

template<int i>
struct ProdFEfunctor {
	static constexpr FEFunction_s obj_comb[9] = {
			// {rank, dim, index, scalar-factor}
	        {0,1,1,0.1}, {0,2,1,0.2}, {0,3,1,0.3},
	        {1,1,1,0.4}, {1,2,1,0.5}, {1,3,1,0.6},
	        {2,1,1,0.7}, {2,2,1,0.8}, {2,3,1,0.9}};

	static void run() {
		prodfe_same_types<FEFunction,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FEDivergence,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FESymmetricGradient,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		//prodfe_same_types<FECurl,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>(); 12. TBD fails..no operator * for it. Analysis pending
		prodfe_same_types<FEGradient,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FELaplacian,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FEDiagonalHessian,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FEHessian,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
	}
};
BOOST_FIXTURE_TEST_CASE(ProdFEObjSameType,FEFixture)
{
	for_<0, 9>::run<ProdFEfunctor>();
}

//// Test case ProdFEObjDiffType
// Type: Positive test case
// Coverage: following classes - ProductFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "*" when types of operands is different but tensor rank is equal
//		This is done by "invoking" operator overloading for several combinations which can conclude that:
// 		- all possible ways to multiply FE functions is tested
//		- result of ProductFEFunction is of correct type
//		There are no check on values since they are available from dealii library function results which we ignore in out test
// Remarks:
// See remark 1 from SumFEObjDiffType
template <typename T>
void prodfe_diff_types(const T& type1, const T& type2)
{
	auto prod1 = type1 * type2;
	auto prod2 = type2 * type1;
#if 0 //See 15
	auto prod3 = prod1 * prod2;
	auto prod4 = prod1 * prod3;
#endif
}

BOOST_AUTO_TEST_CASE(ProdFEObjDiffType){
	const int rank = 2;
	const int dim = 2;
	const int fe_no = 10; //random

	FEFunction<rank, dim, fe_no> F("test_obj");
	auto FG = grad(F);
	auto FGD = div(grad(F));
	//auto FGG = grad(grad(F)); TBD: Compilation fails. Analysis pending see 13
	//auto FGGD = div(grad(grad(F))); TBD: Compilation fails. Analysis pending see 13
	auto FD = div(F);

	prodfe_diff_types(F,F);
	prodfe_diff_types(FG,FG);
	prodfe_diff_types(FD,FD);
 //TBD: Compilation fail, due to missing definition. see point no. 10
#if 0
	//prodfe_diff_types(F,FGD);
	//prodfe_diff_types(FG,FGGD);
	//prodfe_diff_types(FGD,FGD);
	//prodfe_diff_types(FGG,FGG);
	//prodfe_diff_types(FGGD,FGGD);
#endif
}

//TBD. Questions
//1. The construction of FEGradient and FEHessian (atleast) is ambiguous, since we allow construction with or without the
//   function/gradient. i.e. we can create such an object with just a string or with function/gradient. Why so?
//2. The construction of Hessian, Laplacian, Divergence etc do not store the state from the object with which they were created. Why?
//3. In general, why there are no helper functions for FELiftDivergencem FESymmetricGradient, FECurl, FEDiagonalHessian, and should they be added?
//4. There seems to be no way to get to call set_evaluation_flags() function. This needs FEEvaluation object which is not accessible from FEDatas.
//   I wonder if we instead wanted to call the similar function of FEDatas?
//5. No definition of get_curl, discuss
//6. jfi..fixed an issue in line 714 in dealii_matrixfree.h
//7. Remark: I think scalar_factor should be made private with a getter function
//8. Behavior points
//  Following are not possible (due to impossible default assignment operator=). Is this ok from user pov?
//  fe_function_scalar = -fe_function_scalar;
//  fe_function_scalarxx = fe_function_scalar - 10; // not defined operator - in this way
//  fe_function_scalar = fe_function_scalar * 2;
//9. No matching + operator for FECurl. Also, how do we test the result..right now its implicitly verified that the return type is also SumFEFunction
//10. Theoretically, these combinations are possible. Should these be fixed? Both in sum and product
//11. ProductFEFunctions constructor - the static_assertion text should say Product instead of add
//12. No matching * operator for FECurl. Also, how do we test the result..right now its implicitly verified that the return type is also ProdFEFunction
//13. Hessian and div(hessian) fail compilation test
//14. It is currently possible to create a div object initialized with scalar FEFunction. cant check value() due to another error. But should not this be
// refused at compilation time?
//15. Compilation fails under clang as as ambiguous + operator, ok under gcc
