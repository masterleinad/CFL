//////////
#define BOOST_TEST_MODULE TMOD_FEDATA_4_H
#define BOOST_TEST_DYN_LINK
#include "test_fe_data.h"
//////////


//// Test case FEDatasComma
// Type: Positive test case
// Coverage: following classes - FEDatas
// Checks for:
// 1. Object construction
// 2. behavior of overloaded comma operator in this class
//		This is done by "invoking" operator overloading for several combinations which can conclude that:
// 		- all possible ways to adding FEData objects in FEDatas static container is tested
// 3. basic state check for the object in each case


BOOST_FIXTURE_TEST_CASE(FEDatasComma,FEDatasFixture) {

        FEDatas<decltype(fedata_0_system)> fedatas(fedata_0_system);
        BOOST_TEST(fedatas.fe_number == fedatas.get_fe_data<0>().fe_number);

        auto fedatas1((fedata_0_system,fedata_1_system)); //invoking automatic copy construction
        BOOST_TEST(fedatas1.get_fe_data<0>().fe_number == fe_0);
        BOOST_TEST(fedatas1.get_fe_data<1>().fe_number == fe_1);

        //fedatas = (fedatas,fedata_1_system); TBD, cant be supported see point no. 9

        //Check the state of objects added through comma operator
        //FEDatas<decltype(fedata_e_system)> fedatas = fedata_e_system; //fails due to explicit constructor disallowing copy initialization
        auto fedatas2 = (fedatas,fedata_1_system);
        BOOST_TEST(fedatas2.get_fe_data<0>().fe_number == fe_0);
        BOOST_TEST(fedatas2.get_fe_data<1>().fe_number == fe_1);

        auto fedatas3 = (fedatas,fedata_1_system,fedata_2_system);
        BOOST_TEST(fedatas3.get_fe_data<0>().fe_number == fe_0);
        BOOST_TEST(fedatas3.get_fe_data<1>().fe_number == fe_1);
        BOOST_TEST(fedatas3.get_fe_data<2>().fe_number == fe_2);


        auto fedatas4 = (fedata_1_system,fedatas);
        BOOST_TEST(fedatas4.get_fe_data<0>().fe_number == fe_0);
        BOOST_TEST(fedatas4.get_fe_data<1>().fe_number == fe_1);

        auto fedatas5 = (fedatas,fedata_1_system,fedata_2_system);
        BOOST_TEST(fedatas5.get_fe_data<0>().fe_number == fe_0);
        BOOST_TEST(fedatas5.get_fe_data<1>().fe_number == fe_1);
        BOOST_TEST(fedatas5.get_fe_data<2>().fe_number == fe_2);

        //TBD: This fails since we cant combine FEDatas with comma operator. Discuss if needed
#if 0
        auto fedatas_t1 = (fedatas,fedata_0_system,fedata_1_system);
        auto fedatas_t2 = (fedata_2_system,fedata_3_system);
        auto fedatas6 = (fedatas_t1,fedatas_t2);
        BOOST_TEST(fedatas6.get_fe_data<0>().fe_number == fe_0);
        BOOST_TEST(fedatas6.get_fe_data<1>().fe_number == fe_1);
#endif


#if 0
        MatrixFree<dim> data;
        fedatas.initialize<dim>(data);

        //TBD: Add tests for initialize of more fedatas'n' objects
#endif

}
