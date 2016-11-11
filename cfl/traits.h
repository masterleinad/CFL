#ifndef cfl_traits_h
#define cfl_traits_h

namespace CFL
{
namespace Traits
{
  /**
   * \brief The tensor rank and dimension of an object.
   *
   * This structure is used as a public typedef <tt>Traits</tt> in
   * most CFL classes. Thus, matching ranks and dimensions can be
   * checked and operations can be chosen according to these numbers.
   *
   * \todo Should we have different dimensions in different tensor
   * directions?
   */
  template <int rank_, int dim_>
  struct Tensor
  {
    static const unsigned int rank = rank_;
    static const unsigned int dim = dim_;
  };

  /**
   * \brief Indicator for classes with implementation of multiple
   * derivatives in coordinate directions.
   *
   * All expressions can be evaluated with a set of indices according to
   * their tensor rank and dimension. If these expressions involve
   * Gradient or higher derivatives, some of these indices refer to
   * partial derivatives. Gradient does not actually compute the
   * derivative of its argument, but jsut converts the index with
   * respect to the first tensor component into a derivative
   * flag. Terminal classes then have to be able to compute these
   * derivatives, and they do not only evaluate their values, but also
   * derivatives, using a second evaluation function. This evaluation
   * function is calles "simple derivative", as opposed to using product
   * and chain rules etc.
   *
   * This class must be specialized for all terminal expressions, such
   * that hassimple_derivative::value is true for such classes. This
   * property is then inherited by their derivatives and also their
   * sums.
   */
  template <class T>
  struct has_simple_derivative
  {
    static const bool value = false;
  };

  /**
   * \brief Indicator for test functions used in forms
   *
   * This trait is used to ensure that a Form has only one set of test
   * functions and that this set is in first position, such that
   * integrators know where to find it.
   *
   * Test function sets are the exception from the rule, thus most
   * classes have a <tt>false</tt> here.
   */
  template <class T>
  struct is_test_function_set
  {
    static const bool value = false;
  };

  /**
   * \brief True if the object is a CFL unary operator
   */
  template <class T>
  struct is_unary_operator
  {
    static const bool value = false;
  };

  /**
   * \brief True if the object is a CFL binary operator
   */
  template <class T>
  struct is_binary_operator
  {
    static const bool value = false;
  };

  /**
   * \brief The object is already a form consisting of test function
   * set and tested expression
   */
  template <class T>
  struct is_form
  {
    static const bool value = false;
  };

  /**
   * \brief Objectys need binding to mesh objects in loop
   */
  template <class T>
  struct needs_anchor
  {
    static const bool value = false;
  };
}
}

#endif
