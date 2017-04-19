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
 * \brief Interface to the deal.II library
 */
namespace dealii
{
  /**
   * \brief The terminal objects based on dealii::MatrixFree classes.
   */
  namespace MatrixFree
  {
    template <class Derived>
    class TestFunctionBase;

    template <class Derived>
    class FEFunctionBase;

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
  template <typename Number>
  struct is_block_vector<::dealii::LinearAlgebra::distributed::BlockVector<Number>>
  {
    static const bool value = true;
  };

  template <typename Number>
  struct is_block_vector<::dealii::BlockVector<Number>>
  {
    static const bool value = true;
  };

  template <int dim, int rank, typename Number>
  struct is_compatible<::dealii::Tensor<rank, dim, Number>,
                       ::dealii::SymmetricTensor<rank, dim, Number>>
  {
    static const bool value = true;
  };

  template <int dim, int rank, typename Number>
  struct is_compatible<::dealii::SymmetricTensor<rank, dim, Number>,
                       ::dealii::Tensor<rank, dim, Number>>
  {
    static const bool value = true;
  };

  template <typename... Types>
  struct is_cfl_object<dealii::MatrixFree::SumFEFunctions<Types...>>
  {
    static const bool value = true;
  };

  template <class A, typename... Types>
  struct is_summable<A, dealii::MatrixFree::SumFEFunctions<Types...>,
                     typename std::enable_if<is_fe_function_set<A>::value>::type>
  {
    static const bool value = true;
  };

  template <class A, typename... Types>
  struct is_summable<dealii::MatrixFree::SumFEFunctions<Types...>, A,
                     typename std::enable_if<is_fe_function_set<A>::value>::type>
  {
    static const bool value = true;
  };

  template <typename... TypesA, typename... TypesB>
  struct is_summable<dealii::MatrixFree::SumFEFunctions<TypesA...>,
                     dealii::MatrixFree::SumFEFunctions<TypesB...>>
  {
    static const bool value = true;
  };

  template <typename A, typename B>
  struct is_summable<A, B, typename std::enable_if<is_fe_function_set<A>::value &&
                                                   is_fe_function_set<B>::value>::type>
  {
    static const bool value = true;
  };

  template <typename A, typename B>
  struct is_multiplicable<A, B, typename std::enable_if<is_fe_function_set<A>::value &&
                                                        is_fe_function_set<B>::value>::type>
  {
    static const bool value = true;
  };

  template <typename A, typename B>
  struct is_multiplicable<
    A, B, typename std::enable_if_t<std::is_arithmetic<A>() && is_fe_function_set<B>::value>>
  {
    static const bool value = true;
  };

  template <typename A, typename B>
  struct is_multiplicable<
    A, B, typename std::enable_if_t<is_fe_function_set<A>::value && std::is_arithmetic<B>()>>
  {
    static const bool value = true;
  };

  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct is_cfl_object<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::TestFunctionBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const bool value = true;
  };

  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct is_test_function_set<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::TestFunctionBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const bool value = true;
  };

  template <class FEFunctionType>
  struct is_cfl_object<dealii::MatrixFree::FELiftDivergence<FEFunctionType>>
  {
    static const bool value = true;
  };

  template <class FEFunctionType>
  struct is_fe_function_set<dealii::MatrixFree::FELiftDivergence<FEFunctionType>>
  {
    static const bool value = true;
  };

  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct is_cfl_object<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::FEFunctionBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const bool value = true;
  };

  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct is_fe_function_set<
    T<rank, dim, idx>,
    std::enable_if_t<std::is_base_of<dealii::MatrixFree::FEFunctionBase<T<rank, dim, idx>>,
                                     T<rank, dim, idx>>::value>>
  {
    static const bool value = true;
  };

#if 0
  template <typename... Types>
  struct is_fe_function_set<dealii::MatrixFree::ProductFEFunctions<Types...>>
  {
    static const bool value = true;
  };
#endif

  template <typename... Types>
  struct is_fe_function_product<dealii::MatrixFree::ProductFEFunctions<Types...>>
  {
    static const bool value = true;
  };

  template <typename... Types>
  struct is_fe_expr<dealii::MatrixFree::SumFEFunctions<Types...>>
  {
	  static const bool value = true;
  };

  template <typename... Types>
  struct is_fe_expr<dealii::MatrixFree::ProductFEFunctions<Types...>>
  {
	  static const bool value = true;
  };


  template <typename A>
  struct is_fe_expr<A,typename std::enable_if<is_fe_function_set<A>::value>::type>
  {
	  static const bool value = true;
  };

} // namespace Traits

namespace dealii
{
  namespace MatrixFree
  {

    // CRTP
    template <class T>
    class TestFunctionBase
    {
    public:
      // This class should never be constructed
      TestFunctionBase() = delete;
    };

    template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
    class TestFunctionBase<T<rank, dim, idx>>
    {
    public:
      using TensorTraits = Traits::Tensor<rank, dim>;

      static constexpr unsigned int index = idx;
      static constexpr bool scalar_valued = (0 == TensorTraits::rank);
    };

    template <int rank, int dim, unsigned int idx>
    class TestFunction final : public TestFunctionBase<TestFunction<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestFunction<rank, dim, idx>>;
      static constexpr bool integrate_value = true;
      static constexpr bool integrate_gradient = false;

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

    template <int rank, int dim, unsigned int idx>
    class TestDivergence final : public TestFunctionBase<TestDivergence<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestDivergence<rank, dim, idx>>;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = true;

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

    template <int rank, int dim, unsigned int idx>
    class TestSymmetricGradient final
      : public TestFunctionBase<TestSymmetricGradient<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestSymmetricGradient<rank, dim, idx>>;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = true;

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
        std::cout << "submit SymmetricGradient " << Base::index << " " << q << std::endl;
#endif
        phi.template submit_symmetric_gradient<Base::index>(value, q);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class TestCurl final : public TestFunctionBase<TestCurl<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestCurl<rank, dim, idx>>;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = true;

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

    template <int rank, int dim, unsigned int idx>
    class TestGradient final : public TestFunctionBase<TestGradient<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestGradient<rank, dim, idx>>;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = true;

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

    template <int rank, int dim, unsigned int idx>
    class TestHessian final : public TestFunctionBase<TestHessian<rank, dim, idx>>
    {
    public:
      using Base = TestFunctionBase<TestHessian<rank, dim, idx>>;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = false;

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

    template <int rank, int dim, unsigned int idx>
    TestDivergence<rank - 1, dim, idx>
    div(const TestFunction<rank, dim, idx>& /*unused*/)
    {
      return TestDivergence<rank - 1, dim, idx>();
    }

    template <int rank, int dim, unsigned int idx>
    TestGradient<rank + 1, dim, idx>
    grad(const TestFunction<rank, dim, idx>& /*unused*/)
    {
      return TestGradient<rank + 1, dim, idx>();
    }

    template <int rank, int dim, unsigned int idx>
    TestHessian<rank + 1, dim, idx>
    grad(const TestGradient<rank, dim, idx>& /*unused*/)
    {
      return TestHessian<rank + 1, dim, idx>();
    }

    // CRTP
    template <class Derived>
    class FEFunctionBase
    {
    public:
      // This class should never be constructed
      FEFunctionBase() = delete;
    };

    template <template <int, int, unsigned int> class Derived, int rank, int dim, unsigned int idx>
    class FEFunctionBase<Derived<rank, dim, idx>>
    {
    protected:
      const std::string data_name;

    public:
      using TensorTraits = Traits::Tensor<rank, dim>;
      static constexpr unsigned int index = idx;
      double scalar_factor = 1.;

      FEFunctionBase() = delete;

      explicit FEFunctionBase(const std::string name, double new_factor = 1.)
        : data_name(std::move(name))
        , scalar_factor(new_factor)
      {
      }

      explicit FEFunctionBase(double new_factor = 1.)
        : scalar_factor(new_factor)
      {
      }

      const std::string&
      name() const
      {
        return data_name;
      }

      //fefunc * scalar
      template <typename Number>
      typename std::enable_if_t<std::is_arithmetic<Number>::value, Derived<rank, dim, idx>>
      operator*(const Number scalar_factor_) const
      {
        return Derived<rank, dim, idx>(data_name, scalar_factor * scalar_factor_);
      }

      Derived<rank, dim, idx>
      operator-() const
      {
        const Derived<rank, dim, idx> newfunction(-scalar_factor);
        return newfunction;
      }

    };

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
          FEFunctionType(fefunction.name(), fefunction.scalar_value * scalar_factor_));
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunctionType::set_evaluation_flags(phi);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class FESymmetricGradient final : public FEFunctionBase<FESymmetricGradient<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FESymmetricGradient<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_symmetric_gradient<Base::index>(q);
      }

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

    template <int rank, int dim, unsigned int idx>
    class FECurl final : public FEFunctionBase<FESymmetricGradient<rank, dim, idx>>
    {
    public:
      using Base = FEFunctionBase<FESymmetricGradient<rank, dim, idx>>;
      // inherit constructors
      using Base::Base;

      explicit FECurl(const FEFunction<rank - 1, dim, idx>& fefunction)
        : FECurl(fefunction.name(), fefunction.scalar_factor)
      {
      }

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_curl<Base::index>(q);
      }

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

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_gradient<Base::index>(q);
      }

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

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_laplacian<Base::index>(q);
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

      template <class FEDatas>
      auto
      value(const FEDatas& phi, unsigned int q) const
      {
        return Base::scalar_factor * phi.template get_hessian<Base::index>(q);
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

    template <int rank, int dim, unsigned int idx>
    FEGradient<rank + 1, dim, idx>
    grad(const FEFunction<rank, dim, idx>& f)
    {
      return FEGradient<rank + 1, dim, idx>(f);
    }

    template <int rank, int dim, unsigned int idx>
    FEDivergence<rank - 1, dim, idx>
    div(const FEFunction<rank, dim, idx>& f)
    {
      return FEDivergence<rank - 1, dim, idx>(f);
    }

    template <int rank, int dim, unsigned int idx>
    FEHessian<rank + 1, dim, idx>
    grad(const FEGradient<rank, dim, idx>& f)
    {
      return FEHessian<rank + 1, dim, idx>(f);
    }

    template <int rank, int dim, unsigned int idx>
    FELaplacian<rank - 1, dim, idx>
    div(const FEGradient<rank, dim, idx>& f)
    {
      return FELaplacian<rank - 1, dim, idx>(f);
    }

    //number * fefunc
    template <typename Number, class A>
    typename std::enable_if_t<
      CFL::Traits::is_fe_function_set<A>::value && std::is_arithmetic<Number>::value, A>
    operator*(const Number scalar_factor, const A& a)
    {
      return a * scalar_factor;
    }

    template <typename... Types>
    class SumFEFunctions;

    template <class FEExpr>
    class SumFEFunctions<FEExpr>
    {
    public:
      using TensorTraits =
        Traits::Tensor<FEExpr::TensorTraits::rank, FEExpr::TensorTraits::dim>;
      static constexpr unsigned int n=1;

      explicit SumFEFunctions(const FEExpr summand_)
        : summand(std::move(summand_))
      {
        static_assert(Traits::is_fe_expr<FEExpr>::value,
                      "You need to construct this with a FEFunction object!");

      }

#if 0
      //sumfefunc(fefunc) + fefunc
      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
      	  	  	  	  	  	  SumFEFunctions<NewFEFunction, FEFunction>>::type
      operator+(const NewFEFunction& new_summand) const
      {
        static_assert(TensorTraits::dim == NewFEFunction::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == NewFEFunction::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
        return SumFEFunctions<NewFEFunction, FEFunction>(new_summand, summand);
      }

      //sumfe(fefunc) + sumfe(fefunc)
      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
      	  	  	  	  	  	  SumFEFunctions<NewFEFunction, FEFunction>>::type
      operator+(const SumFEFunctions<NewFEFunction>& new_sum) const
      {
        return SumFEFunctions<NewFEFunction, FEFunction>(new_sum.get_summand(), summand);
      }

      //sumfefunc(fefunc) - fefunc
      template <class NewFEFunction>
      SumFEFunctions<NewFEFunction, FEFunction>
      operator-(const NewFEFunction& new_summand) const
      {
        return operator+(-new_summand);
      }

#if 0 //check if really needed
      //sumfe(fefunc) - sumfe(fefunc)
      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
      	  	  	  	  	  	  SumFEFunctions<NewFEFunction, FEFunction>>::type
      operator-(const SumFEFunctions<NewFEFunction>& new_sum) const
      {
    	return operator+(-new_sum);
      }
#endif

      // -sumfefunc
      SumFEFunctions<FEExpr>
      operator-() const
      {
    	  return SumFEFunctions(-summand);
      }
#endif

      // -sumfefunc
      //This looks odd but this is correct because there need not be an
      //independent object of type SumFEFunctions<FEFunction>
      //as this is equivalent to type FEFunction
      FEExpr
      operator-() const
      {
    	  return (-summand);
      }

      template <class FEEvaluation>
      auto
      value(FEEvaluation& phi, unsigned int q) const
      {
        return summand.value(phi, q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
    	  FEExpr::set_evaluation_flags(phi);
      }

      const FEExpr&
      get_summand() const
      {
        return summand;
      }

      //TBD: Discuss with Daniel for keeping this or alternate..after testing
      unsigned int get_fe_func_index(unsigned int order_no) const
      {
    	  //"No FEFunction found at such an index in SumFEFunctions object"
    	  //AssertThrow(order_no == n, ::dealii::StandardExceptions::ExcIndexRange(order_no, 1, n));

    	  return 0; //TBD
    	  //return summand.index; //return FE function index number
      }

    private:
      const FEExpr summand;
    };

    template <class FEExpr, typename... Types>
    class SumFEFunctions<FEExpr, Types...> : public SumFEFunctions<Types...>
    {
    public:
      using TensorTraits =
        Traits::Tensor<FEExpr::TensorTraits::rank, FEExpr::TensorTraits::dim>;
      using Base = SumFEFunctions<Types...>;
      static constexpr unsigned int n=Base::n + 1;

      explicit SumFEFunctions(const FEExpr summand_, const Types... old_sum)
        : Base(std::move(old_sum...))
        , summand(std::move(summand_))
      {
        static_assert(Traits::is_fe_expr<FEExpr>::value,
                      "You need to construct this with a FEFunction object!");
        static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
      }

      SumFEFunctions(const FEExpr& summand_, const SumFEFunctions<Types...>& old_sum)
        : Base(old_sum)
        , summand(summand_)
      {
        static_assert(Traits::is_fe_expr<FEExpr>::value,
                      "You need to construct this with a FEFunction object!");
        static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
      }


      template <class FEEvaluation>
      auto
      value(const FEEvaluation& phi, unsigned int q) const
      {
        const auto own_value = summand.value(phi, q);
        const auto other_value = Base::value(phi, q);
        assert_is_compatible(own_value, other_value);
        return own_value + other_value;
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
    	FEExpr::set_evaluation_flags(phi);
        Base::set_evaluation_flags(phi);
      }

#if 0
      //sumfefunc + fefunc
      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              SumFEFunctions<NewFEFunction, FEFunction, Types...>>::type
      operator+(const NewFEFunction& new_summand) const
      {
        return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_summand, *this);
      }

      //sumfefunc + sumfe(fefunc)
      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              SumFEFunctions<NewFEFunction, FEFunction, Types...>>::type
      operator+(const SumFEFunctions<NewFEFunction>& new_sum) const
      {
        return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_sum.get_summand(), *this);
      }

      //sumfefunc1 + sumfefunc2
      template <class NewFEFunction, typename NewFEFunction2, typename... NewTypes,
                typename = typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,void>::type>
      auto operator+(const SumFEFunctions<NewFEFunction, NewFEFunction2,NewTypes...>& new_sum) const
      {
         	 return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_sum.get_summand(), *this) +
          			SumFEFunctions<NewFEFunction2,NewTypes...>(
          			static_cast<const SumFEFunctions<NewFEFunction2,NewTypes...>&>(new_sum));
      }

      //sumfefunc - fefunc
      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              SumFEFunctions<NewFEFunction, FEFunction, Types...>>::type
      operator-(const NewFEFunction& new_summand) const
      {
        return operator+(-new_summand);
      }

      //sumfefunc1 - sumfefunc2(fefunc)
      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              SumFEFunctions<NewFEFunction, FEFunction, Types...>>::type
      operator-(const SumFEFunctions<NewFEFunction>& new_sum) const
      {
        return operator+(-new_sum);
      }

      //sumfefunc1 - sumfefunc2
      template <class NewFEFunction, typename... NewTypes,
                typename = typename std::enable_if<
                          CFL::Traits::is_fe_function_set<NewFEFunction>::value, void>::type>
      auto operator-(const SumFEFunctions<NewFEFunction, NewTypes...>& new_sum) const
      {
        return operator+(-new_sum);
      }

#endif
      //-sumfefunc
      SumFEFunctions<FEExpr,Types...>
      operator-() const
      {
    	  return (-Base(static_cast<const Base>(*this)) + (-summand));
      }

      const FEExpr&
      get_summand() const
      {
        return summand;
      }
      //TBD: Discuss with Daniel for keeping this or alternate..after testing
      unsigned int get_fe_func_index(unsigned int order_no) const
      {
    	  if (order_no == n)
    	  {
    	  	  //return summand.index; //return FE function index number
    		  return 0; //TBD
    	  }
    	  else
    	  {
    		  return Base::get_fe_func_index(order_no);
    	  }
      }

    private:
      const FEExpr summand;
    };

    //fefunc1 + fefunc2
    //fefunc + sumfefunc and reverse
    //sumfefunc + sumfefunc
    //No not move, we would need cast down in base class, which leads to creation of temporaries and slows down
    //Saurabh - compiles ok
    template <class FEExpr1, class FEExpr2>
    typename std::enable_if<Traits::is_fe_expr<FEExpr1>::value &&
                              Traits::is_fe_expr<FEExpr2>::value,
                            SumFEFunctions<FEExpr2, FEExpr1>>::type
    operator+(const FEExpr1& old_fe_function, const FEExpr2& new_fe_function)
    {

      static_assert(FEExpr1::TensorTraits::dim == FEExpr2::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(FEExpr1::TensorTraits::rank == FEExpr2::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
      return SumFEFunctions<FEExpr2, FEExpr1>(new_fe_function, old_fe_function);
    }

#if 0
    //saurabh - should not be needed anymore
    //For same reason as above, dont move to class
    template <class FEFunction, typename... Types>
    typename std::enable_if<Traits::is_fe_function_set<FEFunction>::value,
                            SumFEFunctions<FEFunction, Types...>>::type
    operator+(const FEFunction& new_fe_function, const SumFEFunctions<Types...>& old_fe_function)
    {
      return old_fe_function + new_fe_function;
    }
#endif

    //fefunc1 - fefunc2 == TBD this should be moved to class
    //For same reason as above, dont move to class
    template <class FEExpr1, class FEExpr2>
    typename std::enable_if<Traits::is_fe_expr<FEExpr1>::value &&
                              Traits::is_fe_expr<FEExpr2>::value,
                            SumFEFunctions<FEExpr2, FEExpr1>>::type
    operator-(const FEExpr1& old_fe_function, const FEExpr2& new_fe_function)
    {
      return old_fe_function + (-new_fe_function);
    }

#if 0
    //fefunc - sumfefunc == TBD, this should be moved to class
    //For same reason as above, dont move to class
    template <class FEExpr, typename... Types>
    typename std::enable_if<Traits::is_fe_expr<FEExpr>::value,
                            SumFEFunctions<FEExpr, Types...>>::type
    operator-(const FEExpr& new_fe_function, const SumFEFunctions<Types...>& old_fe_function)
    {
    	return -old_fe_function + new_fe_function;
    }
#endif

    template <class FEFunction>
    class ProductFEFunctions<FEFunction>
    {
    public:
      using TensorTraits =
        Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;

      explicit ProductFEFunctions(const FEFunction factor_)
        : factor(std::move(factor_))
      {
        static_assert(Traits::is_fe_function_set<FEFunction>::value,
                      "You need to construct this with a FEFunction object!");
      }

      //prodfefunc * fefunc
      template <class NewFEFunction>
      ProductFEFunctions<NewFEFunction, FEFunction> operator*(const NewFEFunction& new_factor) const
      {
        static_assert(Traits::is_fe_function_set<NewFEFunction>::value,
                      "Only FEFunction objects can be added!");
        static_assert(TensorTraits::dim == NewFEFunction::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == NewFEFunction::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
        return ProductFEFunctions<NewFEFunction, FEFunction>(new_factor, factor);
      }

      template <class FEEvaluation>
      auto
      value(FEEvaluation& phi, unsigned int q) const
      {
        return factor.value(phi, q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunction::set_evaluation_flags(phi);
      }

      const FEFunction&
      get_factor() const
      {
        return factor;
      }

    private:
      const FEFunction factor;
    };

    template <class FEFunction, typename... Types>
    class ProductFEFunctions<FEFunction, Types...> : public ProductFEFunctions<Types...>
    {
    public:
      using TensorTraits =
        Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;
      using Base = ProductFEFunctions<Types...>;

      template <class FEEvaluation>
      auto
      value(const FEEvaluation& phi, unsigned int q) const
      {
        const auto own_value = factor.value(phi, q);
        const auto other_value = Base::value(phi, q);
        assert_is_compatible(own_value, other_value);
        return own_value * other_value;
      }

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
        static_assert(Traits::is_fe_function_set<FEFunction>::value,
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
        static_assert(Traits::is_fe_function_set<FEFunction>::value,
                      "You need to construct this with a FEFunction object!");
        static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
      }

      //prodfefunc * fefunc
      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              ProductFEFunctions<NewFEFunction, FEFunction, Types...>>::type
      operator*(const NewFEFunction& new_factor) const
      {
        return ProductFEFunctions<NewFEFunction, FEFunction, Types...>(new_factor, *this);
      }

      template <typename Number>
      typename std::enable_if<std::is_arithmetic<Number>::value,
                              ProductFEFunctions<FEFunction, Types...>>::type
      operator*(const Number scalar_factor) const
      {
        ProductFEFunctions<FEFunction, Types...> tmp = *this;
        tmp.multiply_by_scalar(scalar_factor);
        return tmp;
      }

      template <typename Number>
      std::enable_if_t<std::is_arithmetic<Number>::value>
      multiply_by_scalar(const Number scalar)
      {
        factor.scalar_factor *= scalar;
      }

      //prodfefunc * prodfe(fefunc)
      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              ProductFEFunctions<NewFEFunction, FEFunction, Types...>>::type
      operator*(const ProductFEFunctions<NewFEFunction>& new_product) const
      {
        return ProductFEFunctions<NewFEFunction, FEFunction, Types...>(new_product.get_factor(),
                                                                       *this);
      }

      //prodfefunc1 * prodfefunc2
      template <class NewFEFunction, typename NewFEFunction2, typename... NewTypes>
      typename std::enable_if<
        CFL::Traits::is_fe_function_set<NewFEFunction>::value,
        ProductFEFunctions<NewTypes..., NewFEFunction2,NewFEFunction,FEFunction, Types...>>::type
      operator*(const ProductFEFunctions<NewFEFunction,NewFEFunction2,NewTypes...>& new_product) const
      {
        return ProductFEFunctions<NewFEFunction, FEFunction, Types...>(new_product.get_factor(),
                                                                       *this) *
               ProductFEFunctions<NewFEFunction2,NewTypes...>(
                 static_cast<const ProductFEFunctions<NewFEFunction2,NewTypes...>&>(new_product));
      }

      const FEFunction&
      get_factor() const
      {
        return factor;
      }

    private:
      FEFunction factor;
    };

    //fefunc1 * fefunc2
    template <class FEFunction1, class FEFunction2>
    typename std::enable_if<Traits::is_fe_function_set<FEFunction1>::value &&
                            Traits::is_fe_function_set<FEFunction2>::value,
                            ProductFEFunctions<FEFunction2, FEFunction1>>::type
    operator*(const FEFunction1& old_fe_function, const FEFunction2& new_fe_function)
    {
      static_assert(FEFunction1::TensorTraits::dim == FEFunction2::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(FEFunction1::TensorTraits::rank == FEFunction2::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
      return ProductFEFunctions<FEFunction2, FEFunction1>(new_fe_function, old_fe_function);
    }

    //fefunc1 * prodfefunc
    template <class FEFunction, typename... Types>
    typename std::enable_if<Traits::is_fe_function_set<FEFunction>::value,
                            ProductFEFunctions<FEFunction, Types...>>::type
    operator*(const FEFunction& new_fe_function,
              const ProductFEFunctions<Types...>& old_fe_function)
    {
      return old_fe_function * new_fe_function;
    }

    //number * prodfefunc
    template <typename Number, class A>
    typename std::enable_if_t<
      CFL::Traits::is_fe_function_product<A>::value && std::is_arithmetic<Number>::value, A>
    operator*(const Number scalar_factor, const A& a)
    {
      return a * scalar_factor;
    }
  } // namespace MatrixFree
} // namespace dealii
} // namespace CFL

#endif // CFL_DEALII_MATRIXFREE_H
