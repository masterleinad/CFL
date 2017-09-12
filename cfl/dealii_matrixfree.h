#ifndef cfl_dealii_matrix_free_h
#define cfl_dealii_matrix_free_h

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <cfl/forms.h>
#include <cfl/traits.h>

#include <utility>

#define AssertIndexInRange(index, range)                                                           \
  Assert((index) < (range), ::dealii::ExcIndexRange((index), 0, (range)))

namespace CFL
{
/**
 * @brief Interface to the deal.II library
 */
namespace dealii
{
  /**
   * @brief The terminal objects based on dealii::MatrixFree classes.
   */
  namespace MatrixFree
  {
    template <class Derived>
    class TestFunctionBaseBase;
    template <class Derived>
    class TestFunctionBase;
    template <class Derived>
    class TestFunctionFaceBase;

    template <class Derived>
    class FEFunctionBaseBase;
    template <class Derived>
    class FEFunctionBase;
    template <class Derived>
    class FEFunctionFaceBase;

    template <typename... Types>
    class SumFEFunctions;
    template <typename... Types>
    class ProductFEFunctions;
    template <class FEFunctionType>
    class FELiftDivergence;
  } // namespace MatrixFree
} // namespace dealii

namespace Traits
{
  /**
  * @brief Indicator for type of Block Vector
  *
  * This trait is used to determine if the given type is dealII's parallel
  * distributed vector
  *
  */
  template <typename Number>
  struct is_block_vector<::dealii::LinearAlgebra::distributed::BlockVector<Number>>
  {
    static const bool value = true;
  };

  /**
  * @brief Indicator for type of Block Vector
  * This trait is used to determine if the given type is block vector based on
  * dealII's block vector
  *
  */
  template <typename Number>
  struct is_block_vector<::dealii::BlockVector<Number>>
  {
    static const bool value = true;
  };

  /**
  * @brief Indicator for compatibility of objects of Tensor and SymmetricTensor
  *
  * A trait to determine that a general dealii Tensor and dealii
  * symmetric tensor are of equal dimension, rank and Number so that
  * they can be compatibly used in an expression
  *
  * @note Currently unused
  *
  */
  template <int dim, int rank, typename Number>
  struct is_compatible<::dealii::Tensor<rank, dim, Number>,
                       ::dealii::SymmetricTensor<rank, dim, Number>>
  {
    static const bool value = true;
  };

  /**
  * @brief Indicator for compatibility of objects of Tensor and SymmetricTensor
  * A trait to determine that a general dealii Tensor and dealii
  * symmetric tensor are of equal dimension, rank and Number so that
  * they can be compatibly used in an expression
  *
  * @note Currently unused
  *
  */
  template <int dim, int rank, typename Number>
  struct is_compatible<::dealii::SymmetricTensor<rank, dim, Number>,
                       ::dealii::Tensor<rank, dim, Number>>
  {
    static const bool value = true;
  };

  /**
  * @brief Trait to determine if a given type is CFL SumFEFunctions
  *
  */
  template <typename... Types>
  struct is_cfl_object<dealii::MatrixFree::SumFEFunctions<Types...>>
  {
    static const bool value = true;
  };

  /**
  * @brief Trait to determine if a given type is CFL object
  *
  * Trait to determine if a given type is derived from CFL \ref TestFunctionBaseBase
  *
  */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct is_cfl_object<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::TestFunctionBaseBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const bool value = true;
  };

  /**
   * @brief Trait to determine if a given type is CFL object
   *
   * Trait to determine if a given type is derived from CFL \ref FELiftDivergence
   *
   */
  template <class FEFunctionType>
  struct is_cfl_object<dealii::MatrixFree::FELiftDivergence<FEFunctionType>>
  {
    static const bool value = true;
  };

  /**
   * @brief Trait to determine if a given type is CFL object
   *
   * Trait to determine if a given type is derived from CFL \ref FEFunctionBaseBase
   *
   */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct is_cfl_object<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::FEFunctionBaseBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const bool value = true;
  };

  /**
  * @brief Trait to determine if two FE functions can be summed together
  *
  * This trait is used to check if two FE functions are of same type. i.e.
  * (cell/face) \ref ObjectType so that they can be summed together
  *
  */
  template <typename A, typename B>
  struct is_summable<
    A, B,
    typename std::enable_if<fe_function_set_type<A>::value != ObjectType::none &&
                            fe_function_set_type<A>::value == fe_function_set_type<B>::value>::type>
  {
    static const bool value = true;
  };

  /**
  * @brief Trait to determine if two FE functions can be multiplied together
  *
  * This trait is used to check if two FE functions are of same type. i.e.
  * (cell/face) \ref ObjectType so that they can be multiplied together
  *
  */
  template <typename A, typename B>
  struct is_multiplicable<
    A, B,
    typename std::enable_if<fe_function_set_type<A>::value != ObjectType::none &&
                            fe_function_set_type<A>::value == fe_function_set_type<B>::value>::type>
  {
    static const bool value = true;
  };

  /**
  * @brief Trait to determine if a given constant can be multipled with FE function
  *
  * This trait is used to check if a given constant type is arithmetic so that
  * it can be multiplied with FE function
  *
  */
  template <typename A, typename B>
  struct is_multiplicable<
    A, B, typename std::enable_if_t<std::is_arithmetic<A>() &&
                                    fe_function_set_type<B>::value != ObjectType::none>>
  {
    static const bool value = true;
  };

  /**
  * @brief Trait to determine if a given constant can be multipled with FE function
  *
  * This trait is used to check if a given constant type is arithmetic so that
  * it can be multiplied with FE function
  *
  */
  template <typename A, typename B>
  struct is_multiplicable<
    A, B, typename std::enable_if_t<fe_function_set_type<A>::value != ObjectType::none &&
                                    std::is_arithmetic<B>()>>
  {
    static const bool value = true;
  };

  /**
  * @brief Trait to store measure region as cell for a test function
  *
  * This trait is used to check if the given test function is derived from CFL
  * \ref TestFunctionBase and marks its \ref ObjectType as cell
  *
  */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct test_function_set_type<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::TestFunctionBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  /**
  * @brief Trait to store measure region as face for a test function
  *
  * This trait is used to check if the given test function is derived from CFL
  * \ref TestFunctionFaceBase and marks its \ref ObjectType as face
  *
  */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct test_function_set_type<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::TestFunctionFaceBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const ObjectType value = ObjectType::face;
  };

  /**
  * @brief Trait to store measure region as cell for a \ref FELiftDivergence function
  *
  * This trait is used to check if the given FE function is of type CFL
  * \ref FELiftDivergence and marks its \ref ObjectType as cell
  *
  */
  template <class FEFunctionType>
  struct fe_function_set_type<dealii::MatrixFree::FELiftDivergence<FEFunctionType>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  /**
  * @brief Trait to store measure region as cell type for a FE function
  *
  * This trait is used to check if the given FE function is derived from CFL
  * \ref FEFunctionBase and marks its \ref ObjectType as cell
  *
  */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct fe_function_set_type<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::FEFunctionBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  /**
  * @brief Trait to store measure region as face for a FE function
  *
  * This trait is used to check if the given FE function is derived from CFL
  * \ref FEFunctionFaceBase and marks its \ref ObjectType as face
  *
  */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct fe_function_set_type<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::FEFunctionFaceBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const ObjectType value = ObjectType::face;
  };

  /**
  * @brief Trait to store measure region as face for a \ref SumFEFunctions function
  *
  * This trait is used to mark the \ref ObjectType of an object of type CFL
  * \ref SumFEFunctions as the measure region of its first constituting element
  *
  */
  template <class FirstType, typename... Types>
  struct fe_function_set_type<dealii::MatrixFree::SumFEFunctions<FirstType, Types...>>
  {
    static const ObjectType value = fe_function_set_type<FirstType>::value;
  };

  /**
  * @brief Trait to store measure region as face for a \ref ProductFEFunctions function
  *
  * This trait is used to mark the \ref ObjectType of an object of type CFL
  * \ref ProductFEFunctions as the measure region of its first constituting element
  *
  */
  template <class FirstType, typename... Types>
  struct fe_function_set_type<dealii::MatrixFree::ProductFEFunctions<FirstType, Types...>>
  {
    static const ObjectType value = fe_function_set_type<FirstType>::value;
  };

  /**
  * @brief Determine if a given object is of type CFL \ref ProductFEFunctions
  *
  * Default Trait
  * @todo Should it be removed? Discuss
  */
  template <class T>
  struct is_fe_function_product
  {
    static constexpr bool value = false;
  };

  /**
  * @brief Determine if a given object is of type CFL \ref ProductFEFunctions
  *
  */
  template <typename... Types>
  struct is_fe_function_product<dealii::MatrixFree::ProductFEFunctions<Types...>>
  {
    static const bool value = true;
  };

  /**
  * @brief Determine if a given object is of type CFL \ref SumFEFunctions
  *
  * Default Trait
  * @todo Should it be removed? Discuss
  */
  template <class T>
  struct is_fe_function_sum
  {
    static constexpr bool value = false;
  };

  /**
  * @brief Determine if a given object is of type CFL \ref SumFEFunctions
  *
  */
  template <typename... Types>
  struct is_fe_function_sum<dealii::MatrixFree::SumFEFunctions<Types...>>
  {
    static const bool value = true;
  };
} // namespace Traits

namespace dealii
{
  namespace MatrixFree
  {
    /**
    * @brief TBD
    * \todo Add details
    *
    */
    struct IntegrationFlags
    {
      bool value = false;
      bool value_exterior = false;
      bool gradient = false;
      bool gradient_exterior = false;

      constexpr bool operator&(const IntegrationFlags& other_flags) const
      {
        return (value & other_flags.value) || (value_exterior & other_flags.value_exterior) ||
               (gradient & other_flags.gradient) ||
               (gradient_exterior & other_flags.gradient_exterior);
      }
    };

    // CRTP

    /**
     * Top level base class for Test Functions, should never be constructed
     * Defined for safety reasons
     *
     */
    template <class T>
    class TestFunctionBaseBase
    {
    public:
      // This class should never be constructed
      TestFunctionBaseBase() = delete;
    };

    /**
     * Top level base class for Test Functions
     * See \ref TestFunctionBase for more details
     *
     */
    template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
    class TestFunctionBaseBase<T<rank, dim, idx>>
    {
    public:
      using TensorTraits = Traits::Tensor<rank, dim>;

      static constexpr unsigned int index = idx;
      static constexpr bool scalar_valued = (TensorTraits::rank == 0);
    };

    /**
     * Base class for all Test Function classes
     * @note
     * <li> See that this is a templatized class, with template parameter
     * as the actual derived class. This might look like CRTP pattern, but
     * its not since the base class is not trying to use static polymorphism.
     * This way of implementation allows us to clearly structure our class
     * heirarchy and collect the values of <code> index </code>, and Tensor
     * traits in a single place
     * <li> Also note that because this is a template class, the actual base
     * class which is created after template specialization will be different
     * for each Test Function class. This is different from traditional non-
     * template base-derived heirarchy where all derived classes have common
     * base class.
     */
    template <class Derived>
    class TestFunctionBase : public TestFunctionBaseBase<Derived>
    {
      using TestFunctionBaseBase<Derived>::TestFunctionBaseBase;
    };

    /**
     * Top level base class for Test Functions on Face
     * See \ref TestFunctionBase for more details
     *
     */
    template <class Derived>
    class TestFunctionFaceBase : public TestFunctionBaseBase<Derived>
    {
      using TestFunctionBaseBase<Derived>::TestFunctionBaseBase;
    };

    /**
     * Test Function which provides evaluation on interior faces
     * in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestFunctionInteriorFace final
      : public TestFunctionFaceBase<TestFunctionInteriorFace<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionFaceBase<TestFunctionInteriorFace<rank, dim, idx>>;
      static constexpr const IntegrationFlags integration_flags{ true, false, false, false };

      /**
       * Wrapper around submit_face_value function of FEEvaluation
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestFunction is vector valued or "
                      "the TestFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestFunctionInteriorFace " << Base::index << " " << q << std::endl;
#endif
        phi.template submit_face_value<Base::index, true>(value, q);
      }
    };

    /**
     * Test Function which provides evaluation on exterior faces (boundary)
     * in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestFunctionExteriorFace final
      : public TestFunctionFaceBase<TestFunctionExteriorFace<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionFaceBase<TestFunctionExteriorFace<rank, dim, idx>>;
      static constexpr IntegrationFlags integration_flags{ false, true, false, false };

      /**
       * Wrapper around submit_face_value function of FEEvaluation
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestFunction is vector valued or "
                      "the TestFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestFunctionExteriorFace " << Base::index << " " << q << std::endl;
#endif
        phi.template submit_face_value<Base::index, false>(value, q);
      }
    };

    /**
     * Test Function which provides evaluation of gradients on interior faces
     * in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestNormalGradientInteriorFace final
      : public TestFunctionFaceBase<TestNormalGradientInteriorFace<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionFaceBase<TestNormalGradientInteriorFace<rank, dim, idx>>;
      static constexpr IntegrationFlags integration_flags{ false, false, true, false };

      /**
       * Wrapper around submit_normal_gradient function of FEEvaluation
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestFunction is vector valued or "
                      "the TestFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestNormalGradientInteriorFace " << Base::index << " " << q
                  << std::endl;
#endif
        phi.template submit_normal_gradient<Base::index, true>(value, q);
      }
    };

    /**
     * Test Function which provides evaluation of gradients on exterior faces
     * in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestNormalGradientExteriorFace final
      : public TestFunctionFaceBase<TestNormalGradientExteriorFace<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionFaceBase<TestNormalGradientExteriorFace<rank, dim, idx>>;
      static constexpr IntegrationFlags integration_flags{ false, false, false, true };

      /**
       * Wrapper around submit_normal_gradient function of FEEvaluation
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestFunction is vector valued or "
                      "the TestFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestNormalGradientExteriorFace " << Base::index << " " << q
                  << std::endl;
#endif
        phi.template submit_normal_gradient<Base::index, false>(value, q);
      }
    };

    /**
     * Test Function which provides evaluation on cell in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestFunction final : public TestFunctionBase<TestFunction<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestFunction<rank, dim, idx>>;
      static constexpr IntegrationFlags integration_flags{ true, false, false, false };

      /**
       * Wrapper around submit_value function of FEEvaluation
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestFunction is vector valued or "
                      "the TestFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestFunction " << Base::index << " " << q << std::endl;
#endif
        phi.template submit_value<Base::index>(value, q);
      }
    };

    /**
     * Test Function which provides divergence evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestDivergence final : public TestFunctionBase<TestDivergence<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestDivergence<rank, dim, idx>>;
      static constexpr IntegrationFlags integration_flags{ false, false, true, false };

      /**
       * Wrapper around submit_divergence function of FEEvaluation
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert(FEEvaluation::template rank<Base::index>() > 0,
                      "The proposed FiniteElement has to be "
                      "vector valued for using TestDivergence!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestDivergence " << Base::index << " " << q << std::endl;
#endif
        phi.template submit_divergence<Base::index>(value, q);
      }
    };

    /**
     * Test Function which provides Symmetric Gradient evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestSymmetricGradient final
      : public TestFunctionBase<TestSymmetricGradient<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestSymmetricGradient<rank, dim, idx>>;
      static constexpr IntegrationFlags integration_flags{ false, false, true, false };

      /**
       * Wrapper around submit_symmetric_gradient function of FEEvaluation
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (Base::TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestGradient is vector valued or "
                      "the TestGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestSymmetricGradient " << Base::index << " " << q << std::endl;
#endif
        phi.template submit_symmetric_gradient<Base::index>(value, q);
      }
    };

    /**
     * Test Function which provides curl evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestCurl final : public TestFunctionBase<TestCurl<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestCurl<rank, dim, idx>>;
      static constexpr IntegrationFlags integration_flags{ false, false, true, false };

      /**
       * Wrapper around submit_curl function of FEEvaluation
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestCurl is vector valued or "
                      "the TestCurl is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestCurl " << Base::index << " " << q << std::endl;
#endif
        phi.template submit_curl<Base::index>(value, q);
      }
    };

    /**
     * Test Function which provides gradient evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestGradient final : public TestFunctionBase<TestGradient<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestGradient<rank, dim, idx>>;
      static constexpr IntegrationFlags integration_flags{ false, false, true, false };

      /**
       * Wrapper around submit_gradient function of FEEvaluation
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestGradient is vector valued or "
                      "the TestGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestGradient " << Base::index << " " << q << std::endl;
#endif
        phi.template submit_gradient<Base::index>(value, q);
      }
    };

    /**
     * Test Function which provides hessian evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class TestHessian final : public TestFunctionBase<TestHessian<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestHessian<rank, dim, idx>>;
      static constexpr IntegrationFlags integration_flags{ false, false, false, false };

      /**
       * @todo: Not implemented function
       *
       */
      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& /*phi*/, unsigned int /*q*/, const ValueType& /*value*/)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 2),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestHessian is vector valued or "
                      "the TestHessian is scalar valued and "
                      "the FiniteElement is vector valued!");
        AssertThrow(false, "Not implemented yet!");
      }
    };

    /**
     * Utility function to return a TestDivergence object
     * given a TestFunction
     *
     */
    template <int rank, int dim, unsigned int idx>
    TestDivergence<rank - 1, dim, idx>
    div(const TestFunction<rank, dim, idx>& /*unused*/)
    {
      return TestDivergence<rank - 1, dim, idx>();
    }

    /**
     * Utility function to return a TestGradient object
     * given a TestFunction
     *
     */
    template <int rank, int dim, unsigned int idx>
    TestGradient<rank + 1, dim, idx>
    grad(const TestFunction<rank, dim, idx>& /*unused*/)
    {
      return TestGradient<rank + 1, dim, idx>();
    }

    /**
     * Utility function to return a TestHessian object
     * given a TestGradient
     *
     */
    template <int rank, int dim, unsigned int idx>
    TestHessian<rank + 1, dim, idx>
    grad(const TestGradient<rank, dim, idx>& /*unused*/)
    {
      return TestHessian<rank + 1, dim, idx>();
    }

    // CRTP

    /**
     * Top level base class for FE Functions, should never be constructed
     * Defined for safety reasons
     *
     */
    template <class Derived>
    class FEFunctionBaseBase
    {
    public:
      // This class should never be constructed
      FEFunctionBaseBase() = delete;
    };

    /**
     * Top level base class for FE Functions
     * See \ref FEFunctionBase for more details
     *
     */
    template <template <int, int, unsigned int> class Derived, int rank, int dim, unsigned int idx>
    class FEFunctionBaseBase<Derived<rank, dim, idx>>
    {
    protected:
      const std::string data_name;

    public:
      using TensorTraits = Traits::Tensor<rank, dim>;
      static constexpr unsigned int index = idx;
      double scalar_factor = 1.;

      FEFunctionBaseBase() = delete;

      /**
       * Constructor to optionally allow a string name for FE function
       *
       */
      explicit FEFunctionBaseBase(const std::string name, double new_factor = 1.)
        : data_name(std::move(name))
        , scalar_factor(new_factor)
      {
      }

      /**
       * Default constructor
       *
       */
      explicit FEFunctionBaseBase(double new_factor = 1.)
        : scalar_factor(new_factor)
      {
      }

      /**
       * Returns the name for FE function given during construction
       *
       */
      const std::string&
      name() const
      {
        return data_name;
      }

      /**
       * Allows to scale an FE function with a arithmetic value
       *
       */
      template <typename Number>
      typename std::enable_if_t<std::is_arithmetic<Number>::value, Derived<rank, dim, idx>>
      operator*(const Number scalar_factor_) const
      {
        return Derived<rank, dim, idx>(data_name, scalar_factor * scalar_factor_);
      }

      /**
       * Allows to negate an FE function
       *
       */
      Derived<rank, dim, idx>
      operator-() const
      {
        const Derived<rank, dim, idx> newfunction(-scalar_factor);
        return newfunction;
      }
    };

    /**
     * Base class for all FE Function classes
     * @note
     * <li> See that this is a templatized class, with template parameter
     * as the actual derived class. This might look like CRTP pattern, but
     * its not since the base class is not trying to use static polymorphism.
     * This way of implementation allows us to clearly structure our class
     * heirarchy and collect the values of <code> index </code>, and Tensor
     * traits in a single place
     * <li> Also note that because this is a template class, the actual base
     * class which is created after template specialization will be different
     * for each FE Function class. This is different from traditional non-
     * template base-derived heirarchy where all derived classes have common
     * base class.
     */
    template <class Derived>
    class FEFunctionBase : public FEFunctionBaseBase<Derived>
    {
      using FEFunctionBaseBase<Derived>::FEFunctionBaseBase;
    };

    /**
     * Top level base class for FE Functions on Face
     * See \ref TestFunctionBase for more details
     *
     */
    template <class Derived>
    class FEFunctionFaceBase : public FEFunctionBaseBase<Derived>
    {
      using FEFunctionBaseBase<Derived>::FEFunctionBaseBase;
    };

    /**
     * FE Function which provides evaluation on interior faces
     * in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FEFunctionInteriorFace final
      : public FEFunctionFaceBase<FEFunctionInteriorFace<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionFaceBase<FEFunctionInteriorFace<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        const auto value = Base::scalar_factor * phi.template get_face_value<Base::index, true>(q);
        std::cout << "scalar factor: " << Base::scalar_factor << std::endl;
        return value;
      }

      /**
       * Wrapper around set_evaluation_flags_face function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEFunction is vector valued or "
                      "the FEFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags_face<Base::index>(true, false, false);
      }
    };

    /**
     * FE Function which provides evaluation on exterior faces
     * in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FEFunctionExteriorFace final
      : public FEFunctionFaceBase<FEFunctionExteriorFace<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionFaceBase<FEFunctionExteriorFace<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        const auto value = Base::scalar_factor * phi.template get_face_value<Base::index, false>(q);
        std::cout << "scalar factor: " << Base::scalar_factor << std::endl;
        return value;
      }

      /**
       * Wrapper around set_evaluation_flags_face function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEFunction is vector valued or "
                      "the FEFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags_face<Base::index>(true, false, false);
      }
    };

    /**
     * FE Function which provides gradient evaluation on interior faces
     * in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FENormalGradientInteriorFace final
      : public FEFunctionFaceBase<FENormalGradientInteriorFace<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionFaceBase<FENormalGradientInteriorFace<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        const auto value =
          Base::scalar_factor * phi.template get_normal_gradient<Base::index, true>(q);
        std::cout << "scalar factor: " << Base::scalar_factor << std::endl;
        return value;
      }

      /**
       * Wrapper around set_evaluation_flags function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEFunction is vector valued or "
                      "the FEFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags_face<Base::index>(false, true, false);
      }
    };

    /**
     * FE Function which provides gradient evaluation on exterior faces
     * in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FENormalGradientExteriorFace final
      : public FEFunctionFaceBase<FENormalGradientExteriorFace<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionFaceBase<FENormalGradientExteriorFace<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        const auto value =
          Base::scalar_factor * phi.template get_normal_gradient<Base::index, false>(q);
        std::cout << "scalar factor: " << Base::scalar_factor << std::endl;
        return value;
      }

      /**
       * Wrapper around set_evaluation_flags function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEFunction is vector valued or "
                      "the FEFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags_face<Base::index>(false, true, false);
      }
    };

    /**
     * FE Function which provides evaluation on cell in Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FEFunction final : public FEFunctionBase<FEFunction<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FEFunction<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_value<Base::index>(q);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEFunction is vector valued or "
                      "the FEFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<Base::index>(true, false, false);
      }
    };

    /**
     * FE Function which provides divergence evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FEDivergence final : public FEFunctionBase<FEDivergence<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FEDivergence<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      explicit FEDivergence(const FEFunction<rank + 1, dim, idx>& fefunction)
        : FEDivergence(fefunction.name(), fefunction.scalar_factor)
      {
      }

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_divergence<Base::index>(q);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert(FEEvaluation::template rank<Base::index>() > 0,
                      "The proposed FiniteElement has to be "
                      "vector valued for using FEDivergence!");
        phi.template set_evaluation_flags<Base::index>(false, true, false);
      }
    };

    /**
     * TBD
     *
     */
    template <class FEFunctionType>
    class FELiftDivergence final
    {
    private:
      const FEFunctionType fefunction;

    public:
      using TensorTraits =
        Traits::Tensor<FEFunctionType::TensorTraits::rank + 2, FEFunctionType::TensorTraits::dim>;
      static constexpr unsigned int index = FEFunctionType::idx;

      explicit FELiftDivergence(const FEFunctionType fe_function)
        : fefunction(std::move(fe_function))
      {
      }

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        auto value = fefunction.value(phi, q);
        ::dealii::Tensor<TensorTraits::rank, TensorTraits::dim, decltype(value)> lifted_tensor;
        for (unsigned int i = 0; i < TensorTraits::dim; ++i)
        {
          lifted_tensor[i][i] = value;
        }
        return lifted_tensor;
      }

      auto
      operator-() const
      {
        return FELiftDivergence(-fefunction);
      }

      template <typename Number>
      typename std::enable_if_t<std::is_arithmetic<Number>::value, FELiftDivergence<FEFunctionType>>
      operator*(const Number scalar_factor_) const
      {
        return FELiftDivergence<FEFunctionType>(
          FEFunctionType(fefunction.name(), fefunction.scalar_factor * scalar_factor_));
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunctionType::set_evaluation_flags(phi);
      }
    };

    /**
     * FE Function which provides Symmetric Gradient evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FESymmetricGradient final : public FEFunctionBase<FESymmetricGradient<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FESymmetricGradient<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      /**
       * Wrapper around get_symmetric_gradient function of FEEvaluation
       *
       */
      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_symmetric_gradient<Base::index>(q);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEGradient is vector valued or "
                      "the FEGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<Base::index>(false, true, false);
      }
    };

    /**
     * FE Function which provides curl evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FECurl final : public FEFunctionBase<FECurl<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FECurl<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      explicit FECurl(const FEFunction<rank - 1, dim, idx>& fefunction)
        : FECurl(fefunction.name(), fefunction.scalar_factor)
      {
      }

      /**
       * Wrapper around get_curl function of FEEvaluation
       *
       */
      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_curl<Base::index>(q);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEGradient is vector valued or "
                      "the FEGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<Base::index>(false, true, false);
      }
    };

    /**
     * FE Function which provides gradient evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FEGradient final : public FEFunctionBase<FEGradient<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FEGradient<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      explicit FEGradient(const FEFunction<rank - 1, dim, idx>& fefunction)
        : FEGradient(fefunction.name(), fefunction.scalar_factor)
      {
      }

      /**
       * Wrapper around get_gradient function of FEEvaluation
       *
       */
      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_gradient<Base::index>(q);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEGradient is vector valued or "
                      "the FEGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<Base::index>(false, true, false);
      }
    };

    /**
     * FE Function which provides Laplacian evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FELaplacian final : public FEFunctionBase<FELaplacian<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FELaplacian<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      explicit FELaplacian(const FEGradient<rank + 1, dim, idx>& fe_function)
        : FELaplacian(fe_function.name(), fe_function.scalar_factor)
      {
      }

      /**
       * Wrapper around get_laplacian function of FEEvaluation
       *
       */
      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_laplacian<Base::index>(q);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 2),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEHessian is vector valued or "
                      "the FEHessian is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<Base::index>(false, false, true);
      }
    };

    /**
     * TBD
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FEDiagonalHessian final : public FEFunctionBase<FEDiagonalHessian<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FEDiagonalHessian<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_hessian_diagonal<Base::index>(q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 2),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEHessian is vector valued or "
                      "the FEHessian is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<Base::index>(false, false, true);
      }
    };

    /**
     * FE Function which provides Hessian evaluation on cell in
     * Matrix Free context
     *
     */
    template <int rank, int dim, unsigned int idx>
    class FEHessian final : public FEFunctionBase<FEHessian<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FEHessian<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      explicit FEHessian(const FEGradient<rank - 1, dim, idx>& fefunction)
        : FEHessian(fefunction.name(), fefunction.scalar_factor)
      {
      }

      /**
       * Wrapper around get_hessian function of FEEvaluation
       *
       */
      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_hessian<Base::index>(q);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEEvaluation
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                        (Base::TensorTraits::rank > 2),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEHessian is vector valued or "
                      "the FEHessian is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<Base::index>(false, false, true);
      }
    };

    /**
     * Utility function to return a FEGradient object
     * given a FEFunction
     *
     */
    template <int rank, int dim, unsigned int idx>
    FEGradient<rank + 1, dim, idx>
    grad(const FEFunction<rank, dim, idx>& f)
    {
      return FEGradient<rank + 1, dim, idx>(f);
    }

    /**
     * Utility function to return a FEDivergence object
     * given a FEFunction
     *
     */
    template <int rank, int dim, unsigned int idx>
    FEDivergence<rank - 1, dim, idx>
    div(const FEFunction<rank, dim, idx>& f)
    {
      return FEDivergence<rank - 1, dim, idx>(f);
    }

    /**
     * Utility function to return a FEHessian object
     * given a FEGradient
     *
     */
    template <int rank, int dim, unsigned int idx>
    FEHessian<rank + 1, dim, idx>
    grad(const FEGradient<rank, dim, idx>& f)
    {
      return FEHessian<rank + 1, dim, idx>(f);
    }

    /**
     * Utility function to return a FELaplacian object
     * given a FEGradient
     *
     */
    template <int rank, int dim, unsigned int idx>
    FELaplacian<rank - 1, dim, idx>
    div(const FEGradient<rank, dim, idx>& f)
    {
      return FELaplacian<rank - 1, dim, idx>(f);
    }

    /**
     * Overloading function to scale an FE function with a scalar factor
     *
     */
    template <typename Number, class A>
    typename std::enable_if_t<Traits::fe_function_set_type<A>::value != ObjectType::none &&
                                std::is_arithmetic<Number>::value,
                              A>
    operator*(const Number scalar_factor, const A& a)
    {
      return a * scalar_factor;
    }

    template <typename... Types>
    class SumFEFunctions;

    /**
     * Weak formulation of PDE equations gives rise to forms (linear,bilinear
     * etc). These may consist of several FE functions tested with test Function
     * as a sum. It is possible to define sum/difference of such FE Functions using
     * SumFEFunctions class. The summation is achieved by operator overloading.
     * Difference is summation with a negative sign and can be easily achieved.
     *
     * <h3>Implementation</h3>
     * The summation process does not physically add something. Rather, it
     * maintains a static container (also see \ref FEDatas) of the FE Functions
     * An example would be:
     * <code>
                  auto sum3 = fe_function1 + fe_function1 + fe_function3;
     * </code>
     * This will be stored as:
     * *   @verbatim
     *		SumFEFunctions<FEFunction>  --> holds fe_function1
     *		     ^
     *		     |
     *		     |
     *		SumFEFunctions<FEFunction,FEFunction> --> holds fe_function2
     *		     ^
     *		     |
     *		     |
     *		SumFEFunctions<FEFunction,FEFunction,FEFunction> --> holds fe_function3
     *   @endverbatim
     */
    template <class FEFunction>
    class SumFEFunctions<FEFunction>
    {
    public:
      using TensorTraits =
        Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;

      explicit SumFEFunctions(const FEFunction summand_)
        : summand(std::move(summand_))
      {
        static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                      "You need to construct this with a FEFunction object!");
      }

      /**
       * Operator overloading to create a SumFEFunctions from
       * two FEFunction objects
       *
       */
      template <class NewFEFunction>
      auto
      operator+(const NewFEFunction& new_summand) const
      {
        static_assert(Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none,
                      "Only FEFunction objects can be added!");
        static_assert(TensorTraits::dim == NewFEFunction::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == NewFEFunction::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
        return SumFEFunctions<NewFEFunction, FEFunction>(new_summand, summand);
      }

      /**
       * Operator overloading to create a SumFEFunctions from
       * two FEFunction objects
       *
       */
      template <class NewFEFunction>
      auto
      operator-(const NewFEFunction& new_summand) const
      {
        return operator+(-new_summand);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEFunction
       *
       */
      template <class FEEvaluation>
      auto
      value(const FEEvaluation& phi, unsigned int q) const
      {
        return summand.value(phi, q);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEFunction
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunction::set_evaluation_flags(phi);
      }

      /**
       * Returns the FEFunction object held by this object
       *
       */
      const FEFunction&
      get_summand() const
      {
        return summand;
      }

      /**
       * Unary minus operator overloading to get -SumFEFunctions
       *
       */
      SumFEFunctions<FEFunction>
      operator-() const
      {
        // create a copy
        const SumFEFunctions<FEFunction> copy_this(*this);
        copy_this.multiply_by_scalar(-1.);
        return copy_this;
      }

      /**
       * Scale all FEFunctions of SumFEFunctions by a scalar factor
       *
       */
      template <typename Number>
      typename std::enable_if<std::is_arithmetic<Number>::value, SumFEFunctions<FEFunction>>::type
      operator*(const Number scalar_factor) const
      {
        SumFEFunctions<FEFunction> tmp = *this;
        tmp.multiply_by_scalar(scalar_factor);
        return tmp;
      }

      /**
       * Scale only the FEFunction held by this SumFEFunctions by a scalar factor
       *
       */
      template <typename Number>
      std::enable_if_t<std::is_arithmetic<Number>::value>
      multiply_by_scalar(const Number scalar)
      {
        summand.scalar_factor *= scalar;
      }

    private:
      FEFunction summand;
    };

    /**
    * @brief Class to provide Sum of FE Functions.
    * This is for variadic template definition of the SumFEFunctions class.
    * Please refer to the documentation of the previous class
    */
    template <class FEFunction, typename... Types>
    class SumFEFunctions<FEFunction, Types...> : public SumFEFunctions<Types...>
    {
    public:
      using TensorTraits =
        Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;
      using Base = SumFEFunctions<Types...>;

      /**
       * Actual evaluation of sum of the FE Function values in a Matrix Free
       * context
       *
       */
      template <class FEEvaluation>
      auto
      value(const FEEvaluation& phi, unsigned int q) const
      {
        const auto own_value = summand.value(phi, q);
        const auto other_value = Base::value(phi, q);
        assert_is_compatible(own_value, other_value);
        return own_value + other_value;
      }

      /**
       * Wrapper around set_evaluation_flags of FEFunction
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunction::set_evaluation_flags(phi);
        Base::set_evaluation_flags(phi);
      }

      explicit SumFEFunctions(const FEFunction summand_, const Types... old_sum)
        : Base(std::move(old_sum...))
        , summand(std::move(summand_))
      {
        static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                      "You need to construct this with a FEFunction object!");
        static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
      }

      SumFEFunctions(const FEFunction& summand_, const SumFEFunctions<Types...>& old_sum)
        : Base(old_sum)
        , summand(summand_)
      {
        static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                      "You need to construct this with a FEFunction object!");
        static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
      }

      /**
       * Operator overloading to add FEFunction to existing SumFEFunctions
       *
       */
      template <class NewFEFunction,
                typename std::enable_if<
                  Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction>::value ==
                    Traits::fe_function_set_type<FEFunction>::value &&
                  !Traits::is_fe_function_sum<NewFEFunction>::value>::type* unused = nullptr>
      auto
      operator+(const NewFEFunction& new_summand) const
      {
        return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_summand, *this);
      }

      /**
       * Operator overloading to add two SumFEFunctions
       *
       */
      template <class NewFEFunction1, class NewFEFunction2, typename... NewTypes,
                typename std::enable_if<
                  Traits::fe_function_set_type<NewFEFunction1>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction1>::value ==
                    Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
      auto
      operator+(const SumFEFunctions<NewFEFunction1, NewFEFunction2, NewTypes...>& new_sum) const
      {
        return SumFEFunctions<NewFEFunction1, FEFunction, Types...>(new_sum.get_summand(), *this) +
               SumFEFunctions<NewFEFunction2, NewTypes...>(
                 static_cast<const SumFEFunctions<NewFEFunction2, NewTypes...>&>(new_sum));
      }

      /**
       * Operator overloading to add two SumFEFunctions
       *
       */
      template <class NewFEFunction,
                typename std::enable_if<
                  Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction>::value ==
                    Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
      auto
      operator+(const SumFEFunctions<NewFEFunction>& new_sum) const
      {
        return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_sum.get_summand(), *this);
      }

      /**
       * Operator overloading to subtract a FEFunction from a SumFEFunction
       *
       */
      template <class NewFEFunction,
                typename std::enable_if<
                  Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction>::value ==
                    Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
      auto
      operator-(const NewFEFunction& new_summand) const
      {
        return operator+(-new_summand);
      }

      /**
       * Unary minus operator overloading to get -SumFEFunction
       *
       */
      auto
      operator-() const
      {
        // create a copy
        SumFEFunctions<FEFunction, Types...> copy_this(*this);
        copy_this.multiply_by_scalar(-1.);
        return copy_this;
      }

      /**
       * Multiply all FEFunction objects of this SumFEFunction with a scalar factor
       *
       */
      template <typename Number>
      typename std::enable_if<std::is_arithmetic<Number>::value,
                              SumFEFunctions<FEFunction, Types...>>::type
      operator*(const Number scalar_factor) const
      {
        SumFEFunctions<FEFunction, Types...> tmp = *this;
        tmp.multiply_by_scalar(scalar_factor);
        return tmp;
      }

      /**
       * Multiply only the FEFunction object of this SumFEFunction with a scalar factor
       *
       */
      template <typename Number>
      std::enable_if_t<std::is_arithmetic<Number>::value>
      multiply_by_scalar(const Number scalar)
      {
        summand.scalar_factor *= scalar;
        Base::multiply_by_scalar(scalar);
      }

      /**
       * Operator overloading to subtract a SumFEFuction from a SumFEFunction
       *
       */
      template <class NewFEFunction, typename... NewTypes,
                typename std::enable_if<
                  Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction>::value ==
                    Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
      auto
      operator-(const SumFEFunctions<NewFEFunction, NewTypes...>& new_sum) const
      {
        return operator+(-new_sum);
      }

      /**
       * Operator overloading to subtract a SumFEFuction from a SumFEFunction
       *
       */
      template <class NewFEFunction,
                typename std::enable_if<
                  Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction>::value ==
                    Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
      auto
      operator-(const SumFEFunctions<NewFEFunction>& new_sum) const
      {
        return operator+(-new_sum);
      }

      const FEFunction&
      get_summand() const
      {
        return summand;
      }

    private:
      FEFunction summand;
    };

    /**
     * Operator overloading to add two FEFunction objects to form a SumFEFuction object
     *
     */
    template <class FEFunction1, class FEFunction2,
              typename std::enable_if<
                Traits::fe_function_set_type<FEFunction1>::value != ObjectType::none &&
                Traits::fe_function_set_type<FEFunction1>::value ==
                  Traits::fe_function_set_type<FEFunction2>::value &&
                !Traits::is_fe_function_sum<FEFunction1>::value &&
                !Traits::is_fe_function_sum<FEFunction2>::value>::type* unused = nullptr>
    auto
    operator+(const FEFunction1& old_fe_function, const FEFunction2& new_fe_function)
    {

      static_assert(FEFunction1::TensorTraits::dim == FEFunction2::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(FEFunction1::TensorTraits::rank == FEFunction2::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
      return SumFEFunctions<FEFunction2, FEFunction1>(new_fe_function, old_fe_function);
    }

    /**
     * Operator overloading to add a FEFunction object with a SumFEFuctions object
     *
     */
    template <class FEFunction, typename... Types,
              typename std::enable_if<
                Traits::fe_function_set_type<FEFunction>::value != ObjectType::none &&
                !Traits::is_fe_function_sum<FEFunction>::value>::type* unused = nullptr>
    auto
    operator+(const FEFunction& new_fe_function, const SumFEFunctions<Types...>& old_fe_function)
    {
      return old_fe_function + new_fe_function;
    }

    /**
     * Operator overloading to subtract two FEFunction objects to form a SumFEFuction object
     *
     */
    template <class FEFunction1, class FEFunction2,
              typename std::enable_if<
                Traits::fe_function_set_type<FEFunction1>::value != ObjectType::none &&
                Traits::fe_function_set_type<FEFunction1>::value ==
                  Traits::fe_function_set_type<FEFunction2>::value &&
                !Traits::is_fe_function_sum<FEFunction1>::value &&
                !Traits::is_fe_function_sum<FEFunction2>::value>::type* unused = nullptr>
    auto
    operator-(const FEFunction1& old_fe_function, const FEFunction2& new_fe_function)
    {
      return old_fe_function + (-new_fe_function);
    }

    /**
     * Operator overloading to subtract an FEFunction object from a SumFEFuction object
     *
     */
    template <class FEFunction, typename... Types,
              typename std::enable_if<
                Traits::fe_function_set_type<FEFunction>::value != ObjectType::none &&
                !Traits::is_fe_function_sum<FEFunction>::value>::type* unused = nullptr>
    auto
    operator-(const FEFunction& new_fe_function, const SumFEFunctions<Types...>& old_fe_function)
    {
      return -(old_fe_function - new_fe_function);
    }

    /**
     * See \ref SumFEFunctions.
     * It is possible to define product of such FE Functions using
     * ProductFEFunctions class. The product is achieved by operator overloading.
     *
     * <h3>Implementation</h3>
     * The product process does not physically multiply something. Rather, it
     * maintains a static container (also see \ref FEDatas) of the FE Functions
     * An example would be:
     * <code>
                  auto prod1 = fe_function1 * fe_function1 * fe_function3;
     * </code>
     * This will be stored as:
     * *   @verbatim
     *		ProductFEFunctions<FEFunction>  --> holds fe_function1
     *		     ^
     *		     |
     *		     |
     *		ProductFEFunctions<FEFunction,FEFunction> --> holds fe_function2
     *		     ^
     *		     |
     *		     |
     *		ProductFEFunctions<FEFunction,FEFunction,FEFunction> --> holds fe_function3
     *   @endverbatim
     */
    template <class FEFunction>
    class ProductFEFunctions<FEFunction>
    {
    public:
      using TensorTraits =
        Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;
      static constexpr unsigned int n = 1;

      explicit ProductFEFunctions(const FEFunction factor_)
        : factor(std::move(factor_))
      {
        static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                      "You need to construct this with a FEFunction object!");
      }

      // prodfefunc * fefunc
      /**
       * Operator overloading to multiply a ProductFEFunctions with an FEFunction
       *
       */
      template <class NewFEFunction>
      auto operator*(const NewFEFunction& new_factor) const
      {
        static_assert(Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none,
                      "Only FEFunction objects can be added!");
        static_assert(TensorTraits::dim == NewFEFunction::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == NewFEFunction::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
        return ProductFEFunctions<NewFEFunction, FEFunction>(new_factor, factor);
      }

      /**
       * Wrapper around value function of FEFunction
       *
       */
      template <class FEEvaluation>
      auto
      value(FEEvaluation& phi, unsigned int q) const
      {
        return factor.value(phi, q);
      }

      /**
       * Wrapper around set_evaluation_flags function of FEFunction
       *
       */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunction::set_evaluation_flags(phi);
      }

      /**
       * Returns the FEFunction object held by this object
       *
       */
      const FEFunction&
      get_factor() const
      {
        return factor;
      }

      /**
       * Unary minus operator overloading to get -ProductFEFunctions
       *
       */
      ProductFEFunctions<FEFunction>
      operator-() const
      {
        // create a copy
        ProductFEFunctions<FEFunction> copy_this(*this);
        copy_this.multiply_by_scalar(-1.);
        return copy_this;
      }

      template <typename Number>
      std::enable_if_t<std::is_arithmetic<Number>::value>
      multiply_by_scalar(const Number scalar)
      {
        factor.scalar_factor *= scalar;
      }

    private:
      FEFunction factor;
    };

    /**
    * @brief Class to provide Product of FE Functions.
    * This is for variadic template definition of the ProductFEFunctions class.
    * Please refer to the documentation of the previous class
    */
    template <class FEFunction, typename... Types>
    class ProductFEFunctions<FEFunction, Types...> : public ProductFEFunctions<Types...>
    {
    public:
      using TensorTraits =
        Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;
      using Base = ProductFEFunctions<Types...>;
      static constexpr unsigned int n = Base::n + 1;

      /**
       * Actual evaluation of product of the FE Function values in a Matrix Free
       * context
       *
       */
      template <class FEEvaluation>
      auto
      value(const FEEvaluation& phi, unsigned int q) const
      {
        const auto own_value = factor.value(phi, q);
        const auto other_value = Base::value(phi, q);
        assert_is_compatible(own_value, other_value);
        return own_value * other_value;
      }

      /**
       * Wrapper around set_evaluation_flags of FEFunction
       *
      */
      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunction::set_evaluation_flags(phi);
        Base::set_evaluation_flags(phi);
      }

      explicit ProductFEFunctions(const FEFunction factor_, const Types... old_product)
        : Base(std::move(old_product...))
        , factor(std::move(factor_))
      {
        static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                      "You need to construct this with a FEFunction object!");
        static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
      }

      ProductFEFunctions(const FEFunction factor_, const ProductFEFunctions<Types...> old_product)
        : Base(std::move(old_product))
        , factor(std::move(factor_))
      {
        static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                      "You need to construct this with a FEFunction object!");
        static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
      }

      // prodfefunc * fefunc
      /**
       * Operator overloading to multiply FEFunction to existing ProductFEFunctions
       *
       */
      template <class NewFEFunction,
                typename std::enable_if<
                  Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                    Traits::fe_function_set_type<NewFEFunction>::value ==
                      Traits::fe_function_set_type<FEFunction>::value,
                  ProductFEFunctions<NewFEFunction, FEFunction, Types...>>::type* unused = nullptr>
      auto operator*(const NewFEFunction& new_factor) const
      {
        return ProductFEFunctions<NewFEFunction, FEFunction, Types...>(new_factor, *this);
      }

      /**
       * Unary minus operator overloading to negate a ProductFEFunctions
       *
       */
      ProductFEFunctions<FEFunction, Types...>
      operator-() const
      {
        // create a copy
        ProductFEFunctions<FEFunction, Types...> copy_this(*this);
        copy_this.multiply_by_scalar(-1.);
        return copy_this;
      }

      // prodfefunc * number
      /**
       * Multiply all components of ProductFEFunctions with a scalar factor
       *
       */
      template <typename Number,
                typename std::enable_if<std::is_arithmetic<Number>::value>::type* = nullptr>
      auto operator*(const Number scalar_factor) const
      {
        ProductFEFunctions<FEFunction, Types...> tmp = *this;
        tmp.multiply_by_scalar(scalar_factor);
        return tmp;
      }

      /**
       * Multiply only the component held by this ProductFEFunctions with a scalar factor
       *
       */
      template <typename Number>
      std::enable_if_t<std::is_arithmetic<Number>::value>
      multiply_by_scalar(const Number scalar)
      {
        factor.scalar_factor *= scalar;
      }

      // prodfefunc * prodfefunc
      /**
       * Operator overloading to multiply two ProductFEFunctions
       *
       */
      template <class NewFEFunction1, class NewFEFunction2, typename... NewTypes,
                typename std::enable_if<
                  Traits::fe_function_set_type<NewFEFunction1>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction1>::value ==
                    Traits::fe_function_set_type<FEFunction>::value &&
                  Traits::fe_function_set_type<NewFEFunction2>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction2>::value ==
                    Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
      auto operator*(
        const ProductFEFunctions<NewFEFunction1, NewFEFunction2, NewTypes...>& new_product) const
      {
        return ProductFEFunctions<NewFEFunction1, FEFunction, Types...>(new_product.get_factor(),
                                                                        *this) *
               ProductFEFunctions<NewFEFunction2, NewTypes...>(
                 static_cast<const ProductFEFunctions<NewFEFunction2, NewTypes...>&>(new_product));
      }

      // prodfefunc * prodfefunc
      /**
       * Operator overloading to multiply two ProductFEFunctions
       *
       */
      template <class NewFEFunction,
                typename std::enable_if<
                  Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction>::value ==
                    Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
      auto operator*(const ProductFEFunctions<NewFEFunction>& new_product) const
      {
        return ProductFEFunctions<NewFEFunction, FEFunction, Types...>(new_product.get_factor(),
                                                                       *this);
      }

      const FEFunction&
      get_factor() const
      {
        return factor;
      }

    private:
      FEFunction factor;
    };

    // fefunc * fefunc
    /**
     * Operator overloading to multiply two FEFunction objects to get a
     * ProductFEFunctions
     *
     */
    template <class FEFunction1, class FEFunction2,
              typename std::enable_if<
                Traits::fe_function_set_type<FEFunction1>::value != ObjectType::none &&
                  Traits::fe_function_set_type<FEFunction1>::value ==
                    Traits::fe_function_set_type<FEFunction2>::value &&
                  !Traits::is_fe_function_product<FEFunction1>::value &&
                  !Traits::is_fe_function_product<FEFunction2>::value,
                ProductFEFunctions<FEFunction2, FEFunction1>>::type* unused = nullptr>
    auto operator*(const FEFunction1& old_fe_function, const FEFunction2& new_fe_function)
    {
      static_assert(FEFunction1::TensorTraits::dim == FEFunction2::TensorTraits::dim,
                    "You can only multiply tensors of equal dimension!");
      static_assert(FEFunction1::TensorTraits::rank == FEFunction2::TensorTraits::rank,
                    "You can only multiply tensors of equal rank!");
      return ProductFEFunctions<FEFunction2, FEFunction1>(new_fe_function, old_fe_function);
    }

    // Note: this function changes the order of template parameters compared with its other sibling
    // functions which perform multiplication operation on prodfefunc and fefunc
    // fefunc * prodfefunc
    /**
     * Operator overloading to multiply an FEFunction object to a
     * ProductFEFunctions
     *
     */
    template <
      class FEFunction, typename... Types,
      typename std::enable_if<Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                              ProductFEFunctions<FEFunction, Types...>>::type* unused = nullptr>
    auto operator*(const FEFunction& new_fe_function,
                   const ProductFEFunctions<Types...>& old_fe_function)
    {
      return old_fe_function * new_fe_function;
    }
  } // namespace MatrixFree
} // namespace dealii
} // namespace CFL

#endif // CFL_DEALII_MATRIXFREE_H
