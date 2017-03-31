///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_1_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case TestObjCreation
// Type: Positive test case
// Coverage: following classes - TestFunction, TestDivergence, TestSymmetricGradient, TestCurl, TestGradient, TestHessian
// Checks for:
// 1. Template instantiation with several combinations of (rank,dim,index) as per  variable obj_comb
// 2. Object creation in each case, and basic state check for the object
template<int i>
struct TestDatafunctor {
	static constexpr TestData_s obj_comb[9] = {
			// {rank, dim, index, scalar_valued}
	        {0,1,1,1}, {0,2,1,1}, {0,3,1,1},
	        {1,1,1,0}, {1,2,1,0}, {1,3,1,0},
	        {2,1,1,0}, {2,2,1,0}, {2,3,1,0}};
	typedef map<string, pair<bool,bool>> comb_map_t; //Testxxx class name, (integrate_value, integrate_gradient)
	static  comb_map_t i_comb;

	static void run() {
				//TBD: Change to avoid duplication??
        		TestFunction<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj1;
        		//Check basic state of object
        		BOOST_TEST(test_obj1.index == obj_comb[i].index);
        		BOOST_TEST(test_obj1.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj1.integrate_value == i_comb["TestFunction"].first);
        		BOOST_TEST(test_obj1.integrate_gradient == i_comb["TestFunction"].second);

        		TestDivergence<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj2;
        		//Check basic state of object
        		BOOST_TEST(test_obj2.index == obj_comb[i].index);
        		BOOST_TEST(test_obj2.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj2.integrate_value == i_comb["TestDivergence"].first);
        		BOOST_TEST(test_obj2.integrate_gradient == i_comb["TestDivergence"].second);

        		TestSymmetricGradient<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj3;
        		//Check basic state of object
        		BOOST_TEST(test_obj3.index == obj_comb[i].index);
        		BOOST_TEST(test_obj3.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj3.integrate_value == i_comb["TestSymmetricGradient"].first);
        		BOOST_TEST(test_obj3.integrate_gradient == i_comb["TestSymmetricGradient"].second);

        		TestCurl<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj4;
        		//Check basic state of object
        		BOOST_TEST(test_obj4.index == obj_comb[i].index);
        		BOOST_TEST(test_obj4.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj4.integrate_value == i_comb["TestCurl"].first);
        		BOOST_TEST(test_obj4.integrate_gradient == i_comb["TestCurl"].second);

        		TestGradient<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj5;
        		//Check basic state of object
        		BOOST_TEST(test_obj5.index == obj_comb[i].index);
        		BOOST_TEST(test_obj5.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj5.integrate_value == i_comb["TestGradient"].first);
        		BOOST_TEST(test_obj5.integrate_gradient == i_comb["TestGradient"].second);

        		TestHessian<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj6;
        		//Check basic state of object
        		BOOST_TEST(test_obj6.index == obj_comb[i].index);
        		BOOST_TEST(test_obj6.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj6.integrate_value == i_comb["TestHessian"].first);
        		BOOST_TEST(test_obj6.integrate_gradient == i_comb["TestHessian"].second);
        }
};

//ugly initialization of static member
template<int i> typename TestDatafunctor<i>::comb_map_t TestDatafunctor<i>::i_comb = {
		{"TestFunction", {true,false}},
		{"TestDivergence", {false,true}},
		{"TestSymmetricGradient", {false,true}},
		{"TestCurl", {false,true}},
		{"TestGradient", {false,true}},
		{"TestHessian", {false,false}}};

BOOST_AUTO_TEST_CASE(TestObjCreation) {
	//Check some combinations of Test objects
    for_<0, 3>::run<TestDatafunctor>();
}

////////////////////////////////////////////////////////////////////////////



