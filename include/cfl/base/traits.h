#ifndef cfl_traits_h
#define cfl_traits_h

#include <type_traits>

namespace CFL
{

enum struct ObjectType
{
  none,
  cell,
  face
};

namespace Traits
{
  template <class VectorType>
  struct is_block_vector
  {
    static constexpr bool value = false;
  };

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
    static constexpr unsigned int rank = rank_;
    static constexpr unsigned int dim = dim_;
  };

  /**
   * \brief Indicator for classes with implementation of multiple
   * derivatives in coordinate directions.
   *
   * All expressions can be evaluated with a set of indices according to
   * their tensor rank and dimension. If these expressions involve
   * Gradient or higher derivatives, some of these indices refer to
   * partial derivatives. Gradient does not actually compute the
   * derivative of its argument, but just converts the index with
   * respect to the first tensor component into a derivative
   * flag. Terminal classes then have to be able to compute these
   * derivatives, and they do not only evaluate their values, but also
   * derivatives, using a second evaluation function. This evaluation
   * function is called "simple derivative", as opposed to using product
   * and chain rules etc.
   *
   * This class must be specialized for all terminal expressions, such
   * that has simple_derivative::value is true for such classes. This
   * property is then inherited by their derivatives and also their
   * sums.
   */
  template <class T>
  struct has_simple_derivative
  {
    static constexpr bool value = false;
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
  template <class T, class Enable = void>
  struct test_function_set_type
  {
    static constexpr ObjectType value = ObjectType::none;
  };

  template <class T, class Enable = void>
  struct is_cfl_object
  {
    static constexpr bool value = false;
  };

  template <class T>
  struct is_terminal_string
  {
    static constexpr bool value = false;
  };

  template <class T, class U, class Enable = void>
  struct is_summable
  {
    static constexpr bool value = false;
  };

  template <class T, class U, class Enable = void>
  struct is_multiplicable
  {
    static constexpr bool value = false;
  };

  template <class A, class B>
  struct is_compatible
  {
    static constexpr bool value = false;
  };

  template <class A>
  struct is_compatible<A, A>
  {
    static constexpr bool value = true;
  };

  template <class T>
  struct is_fe_data
  {
    static constexpr bool value = false;
  };

  template <class T>
  struct is_fe_data_face
  {
    static constexpr bool value = false;
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
  template <class T, class Enable = void>
  struct fe_function_set_type
  {
    static constexpr ObjectType value = ObjectType::none;
  };

  /**
   * \brief True if the object is a CFL unary operator
   */
  template <class T>
  struct is_unary_operator
  {
    static constexpr bool value = false;
  };

  /**
   * \brief True if the object is a CFL binary operator
   */
  template <class T>
  struct is_binary_operator
  {
    static constexpr bool value = false;
  };

  /**
   * \brief The object is already a form consisting of test function
   * set and tested expression
   */
  template <class T>
  struct is_form
  {
    static constexpr bool value = false;
  };

  /**
   * \brief Objectys need binding to mesh objects in loop
   */
  template <class T>
  struct needs_anchor
  {
    static constexpr bool value = false;
  };
} // namespace Traits

template <class A, class B>
void
assert_is_summable(const A& /*unused*/, const B& /*unused*/)
{
  static_assert(CFL::Traits::is_summable<A, B>::value, "The sum of these objects is not defined!");
}

template <class A, class B>
void
assert_is_multiplicable(const A& /*unused*/, const B& /*unused*/)
{
  static_assert(CFL::Traits::is_multiplicable<A, B>::value,
                "The product of these objects is not defined!");
}

template <class A, class B>
void
assert_is_compatible(const A& /*unused*/, const B& /*unused*/)
{
  static_assert(CFL::Traits::is_compatible<A, B>::value,
                "Types must be compatible to be added! "
                "Probably, the ansatz functions don't match the bilinear form!");
}
} // namespace CFL

#endif
