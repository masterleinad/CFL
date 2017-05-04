//////////
#define BOOST_TEST_MODULE TMOD_FEDATA_3_H
#define BOOST_TEST_DYN_LINK
#include "test_fe_data.h"
//////////

//// Test case FEDataNegative
// Type: Negative test case
// Coverage: following classes - FEData and FEDataFace
// Checks for:
//  1. Runtime exceptions
BOOST_AUTO_TEST_CASE(FEDataNegative)
{
  FE_Q<2> fe(2);

  // Mismatch in FE degree
  typedef FEData<FE_Q, 3 /*unequal degree*/, 1, 2, 1, 4> ErrFEData1;
  BOOST_CHECK_THROW(ErrFEData1 obj(fe), dealii::ExcIndexRange);

  typedef FEDataFace<FE_Q, 3 /*unequal degree*/, 1, 2, 1, 4> ErrFEDataFace1;
  BOOST_CHECK_THROW(ErrFEDataFace1 obj(fe), dealii::ExcIndexRange);


  // Mmismatch in number of components
  typedef FEData<FE_Q, 2, 2 /*unequal components*/, 2, 1, 4> ErrFEData2;
  BOOST_CHECK_THROW(ErrFEData2 obj(fe), dealii::ExcDimensionMismatch);

  typedef FEDataFace<FE_Q, 2, 2 /*unequal components*/, 2, 1, 4> ErrFEDataFace2;
  BOOST_CHECK_THROW(ErrFEDataFace2 obj(fe), dealii::ExcDimensionMismatch);

}
