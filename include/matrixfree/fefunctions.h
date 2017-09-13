#ifndef cfl_dealii_matrix_free_h
#define cfl_dealii_matrix_free_h

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <cfl/fefunctions.h>
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
namespace dealii::MatrixFree
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

  template <class FEFunctionType>
  class FELiftDivergence;
} // namespace MatrixFree

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

    /**
     * Top level base class for Test Functions, should never be constructed
     * Defined for safety reasons
     *
     */
    template <class T>
    class TestFunctionBaseBase;

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

    template <auto... ints>
    auto
    transform(const Base::TestFunctionInteriorFace<ints...>&)
    {
      return TestFunctionInteriorFace<ints...>();
    }
    template <auto... ints>
    auto
    transform(const Base::TestFunctionExteriorFace<ints...>&)
    {
      return TestFunctionExteriorFace<ints...>();
    }
    template <auto... ints>
    auto
    transform(const Base::TestNormalGradientInteriorFace<ints...>&)
    {
      return TestNormalGradientInteriorFace<ints...>();
    }
    template <auto... ints>
    auto
    transform(const Base::TestNormalGradientExteriorFace<ints...>&)
    {
      return TestNormalGradientExteriorFace<ints...>();
    }
    template <auto... ints>
    auto
    transform(const Base::TestFunction<ints...>&)
    {
      return TestFunction<ints...>();
    }
    template <auto... ints>
    auto
    transform(const Base::TestDivergence<ints...>&)
    {
      return TestDivergence<ints...>();
    }
    template <auto... ints>
    auto
    transform(const Base::TestSymmetricGradient<ints...>&)
    {
      return TestSymmetricGradient<ints...>();
    }
    template <auto... ints>
    auto
    transform(const Base::TestCurl<ints...>&)
    {
      return TestCurl<ints...>();
    }
    template <auto... ints>
    auto
    transform(const Base::TestGradient<ints...>&)
    {
      return TestGradient<ints...>();
    }
    template <auto... ints>
    auto
    transform(const Base::TestHessian<ints...>&)
    {
      return TestHessian<ints...>();
    }

    // don't transform things that have not been specified
    /*template <class Type>
    auto
    transform(Type &&f)
    {
      return std::forward<Type>(f);
    }*/

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
    public:
      using TensorTraits = Traits::Tensor<rank, dim>;
      static constexpr unsigned int index = idx;
      const double scalar_factor = 1.;

      /**
       * Default constructor
       *
       */
      constexpr explicit FEFunctionBaseBase(double new_factor = 1.)
        : scalar_factor(new_factor)
      {
      }

      /**
       * Allows to scale an FE function with a arithmetic value
       *
       */
      template <typename Number>
      typename std::enable_if_t<std::is_arithmetic<Number>::value, Derived<rank, dim, idx>>
      operator*(const Number scalar_factor_) const
      {
        return Derived<rank, dim, idx>(scalar_factor * scalar_factor_);
      }

      /**
       * Allows to negate an FE function
       *
       */
      Derived<rank, dim, idx>
      operator-() const
      {
        const Derived<rank, dim, idx> new_function(-scalar_factor);
        return new_function;
      }
    };

    /**
     * Base class for all FE Function classes
     * @note
     * <li> See that this is a templatized class, with template parameter
     * as the actual derived class. This might look like CRTP pattern, but
     * its not since the base class is not trying to use static polymorphism.
     * This way of implementation allows us to clearly structure our class
     * hierarchy and collect the values of <code> index </code>, and Tensor
     * traits in a single place
     * <li> Also note that because this is a template class, the actual base
     * class which is created after template specialization will be different
     * for each FE Function class. This is different from traditional non-
     * template base-derived hierarchy where all derived classes have common
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

    template <auto... ints>
    auto
    transform(const Base::FEFunctionInteriorFace<ints...>& f)
    {
      return FEFunctionInteriorFace<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FEFunctionExteriorFace<ints...>& f)
    {
      return FEFunctionExteriorFace<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FENormalGradientInteriorFace<ints...>& f)
    {
      return FENormalGradientInteriorFace<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FENormalGradientExteriorFace<ints...>& f)
    {
      return FENormalGradientExteriorFace<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FEFunction<ints...>& f)
    {
      return FEFunction<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FEDivergence<ints...>& f)
    {
      return FEDivergence<ints...>(f.scalar_factor);
    }
    template <class Type>
    auto
    transform(const Base::FELiftDivergence<Type>& f)
    {
      return FELiftDivergence<decltype(transform(std::declval<Type>()))>(
        transform(f.get_fefunction()));
    }
    template <auto... ints>
    auto
    transform(const Base::FESymmetricGradient<ints...>& f)
    {
      return FESymmetricGradient<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FECurl<ints...>& f)
    {
      return FECurl<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FEGradient<ints...>& f)
    {
      return FEGradient<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FEDiagonalHessian<ints...>& f)
    {
      return FEDiagonalHessian<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FEHessian<ints...>& f)
    {
      return FEHessian<ints...>(f.scalar_factor);
    }
    template <auto... ints>
    auto
    transform(const Base::FELaplacian<ints...>& f)
    {
      return FELaplacian<ints...>(f.scalar_factor);
    }

    template <class... Types>
    auto transform(const Base::SumFEFunctions<Types...>& f);

    template <class... Types>
    auto transform(const Base::ProductFEFunctions<Types...>& f);

    template <class... Types>
    auto
    transform(const Base::SumFEFunctions<Types...>& f)
    {
      return Base::SumFEFunctions<decltype(transform(std::declval<Types>()))...>(f);
    }

    template <class... Types>
    auto
    transform(const Base::ProductFEFunctions<Types...>& f)
    {
      return Base::ProductFEFunctions<decltype(transform(std::declval<Types>()))...>(f);
    }

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

      template <class OtherFEFunctionType>
      explicit FELiftDivergence(const Base::FELiftDivergence<OtherFEFunctionType>& other_function)
        : fefunction(transform(other_function.get_fefunction()))
      {
      }

      explicit FELiftDivergence(FEFunctionType fe_function)
        : fefunction(std::move(fe_function))
      {
      }

      const FEFunctionType&
      get_fefunction()
      {
        return fefunction;
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
  } // namespace MatrixFree
} // namespace dealii
} // namespace CFL

#endif // CFL_DEALII_FEFUNCTIONS_H
