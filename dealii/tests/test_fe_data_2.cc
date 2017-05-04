//////////
#define BOOST_TEST_MODULE TMOD_FEDATA_2_H
#define BOOST_TEST_DYN_LINK
#include "test_fe_data.h"
//////////

//// Test case FEDataPositive
// Type: Positive test case
// Coverage: following classes - FEData and FEDataFace
// Checks for:
// 1. Object construction
// 2. behavior of overloaded comma operator in this class
//      - fedata,fedata and fedataface,fedataface
//      - fedata,fedatas and fedataface,fedatas
//      - mix of fedata and fedataface
// 3. basic state check for the object in each case
BOOST_AUTO_TEST_CASE(FEDataPositive)
{

  // Check that all shared pointers point to a common managed object
  const auto fe_shared = std::make_shared<FE_Q<2>>(2);
  BOOST_TEST(fe_shared.use_count() == 1);
  FEData<FE_Q, 2, 1, 2, 0, 2, double> fedata_e_system(fe_shared);
  BOOST_TEST(fe_shared.use_count() == 2);
  FEData<FE_Q, 2, 1, 2, 1, 2, double> fedata_u_system(fe_shared);
  BOOST_TEST(fe_shared.use_count() == 3);
  FEData<FE_Q, 2, 1, 2, 2, 2, double> fedata_x_system(fe_shared);
  BOOST_TEST(fe_shared.use_count() == 4);

  FEDataFace<FE_Q, 2, 1, 2, 0, 2, double> feface_e_system(fe_shared);
  BOOST_TEST(fe_shared.use_count() == 5);
  FEDataFace<FE_Q, 2, 1, 2, 1, 2, double> feface_u_system(fe_shared);
  BOOST_TEST(fe_shared.use_count() == 6);
  FEDataFace<FE_Q, 2, 1, 2, 2, 2, double> feface_x_system(fe_shared);
  BOOST_TEST(fe_shared.use_count() == 7);

  // Check the state of objects added through comma operator
  // FEData only
  fedata_e_system, fedata_u_system; // This is just to check that comma operation is not broken
  auto fedatas_2 = (fedata_e_system, fedata_u_system); // ok, but using () has to be remembered
  BOOST_TEST(fedatas_2.get_fe_data<0>().fe_number == fe_0);
  BOOST_TEST(fedatas_2.get_fe_data<1>().fe_number == fe_1);
  auto fedatas_3 =
    (fedata_e_system, fedata_u_system, fedata_x_system); // ok, but using () has to be remembered
  BOOST_TEST(fedatas_3.get_fe_data<0>().fe_number == fe_0);
  BOOST_TEST(fedatas_3.get_fe_data<1>().fe_number == fe_1);
  BOOST_TEST(fedatas_3.get_fe_data<2>().fe_number == fe_2);

  // FEDataFace only
  feface_e_system, feface_u_system; // This is just to check that comma operation is not broken
  auto fefaces_2 = (feface_e_system, feface_u_system); // ok, but using () has to be remembered
  BOOST_TEST(fefaces_2.get_fe_data_face<0>().fe_number == fe_0);
  BOOST_TEST(fefaces_2.get_fe_data_face<1>().fe_number == fe_1);
  auto fefaces_3 =
    (feface_e_system, feface_u_system, feface_x_system); // ok, but using () has to be remembered
  BOOST_TEST(fefaces_3.get_fe_data_face<0>().fe_number == fe_0);
  BOOST_TEST(fefaces_3.get_fe_data_face<1>().fe_number == fe_1);
  BOOST_TEST(fefaces_3.get_fe_data_face<2>().fe_number == fe_2);

  // Mix of FEData and FEDataFace
  auto femix_2 = (feface_u_system, fedata_e_system);
  BOOST_TEST(femix_2.get_fe_data_face<1>().fe_number == fe_1);
  BOOST_TEST(femix_2.get_fe_data<0>().fe_number == fe_0);

  auto femix_3 = (fedata_e_system, feface_u_system);
  BOOST_TEST(femix_3.get_fe_data_face<1>().fe_number == fe_1);
  BOOST_TEST(femix_3.get_fe_data<0>().fe_number == fe_0);
}
