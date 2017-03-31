//////////
#define BOOST_TEST_MODULE TMOD_FEDATA_2_H
#define BOOST_TEST_DYN_LINK
#include "test_fe_data.h"
//////////



//// Test case FEDataPositive
// Type: Positive test case
// Coverage: following classes - FEData
// Checks for:
// 1. Object construction
// 2. behavior of overloaded comma operator in this class
// 3. basic state check for the object in each case
BOOST_AUTO_TEST_CASE(FEDataPositive) {

		//Check that all shared pointers point to a common managed object
        const auto fe_shared = std::make_shared<FE_Q<2>>(2);
        BOOST_TEST(fe_shared.use_count() == 1);
        FEData<FE_Q, 2, 1, 2, 0, 2, double> fedata_e_system(fe_shared);
        BOOST_TEST(fe_shared.use_count() == 2);
        FEData<FE_Q, 2, 1, 2, 1, 2, double> fedata_u_system(fe_shared);
        BOOST_TEST(fe_shared.use_count() == 3);
        FEData<FE_Q, 2, 1, 2, 2, 2, double> fedata_x_system(fe_shared);
        BOOST_TEST(fe_shared.use_count() == 4);

        //Check the state of objects added through comma operator
        //fedata_e_system,fedata_u_system; //ok..but this is strange, maybe it can just be ignored
        auto fedatas_2 = (fedata_e_system,fedata_u_system); //ok, but using () has to be remembered
        BOOST_TEST(fedatas_2.get_fe_data<0>().fe_number == fe_0);
        BOOST_TEST(fedatas_2.get_fe_data<1>().fe_number == fe_1);
        auto fedatas_3 = (fedata_e_system,fedata_u_system,fedata_x_system); //ok, but using () has to be remembered
        BOOST_TEST(fedatas_3.get_fe_data<0>().fe_number == fe_0);
        BOOST_TEST(fedatas_3.get_fe_data<1>().fe_number == fe_1);
        BOOST_TEST(fedatas_3.get_fe_data<2>().fe_number == fe_2);

}
