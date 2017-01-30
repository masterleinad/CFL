#ifndef cfl_dealii_matrix_free_h
#define cfl_dealii_matrix_free_h

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <cfl/forms.h>
#include <cfl/traits.h>

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
    template <int rank, int dim, unsigned int idx>
    class TestFunction;
    template <int rank, int dim, unsigned int idx>
    class TestDivergence;
    template <int rank, int dim, unsigned int idx>
    class TestCurl;
    template <int rank, int dim, unsigned int idx>
    class TestSymmetricGradient;
    template <int rank, int dim, unsigned int idx>
    class TestGradient;
    template <int rank, int dim, unsigned int idx>
    class TestHessian;

    template <int rank, int dim, unsigned int idx>
    class FEFunction;
    template <int rank, int dim, unsigned int idx>
    class FECurl;
    template <int rank, int dim, unsigned int idx>
    class FEDivergence;
    template <int rank, int dim, unsigned int idx>
    class FESymmetricGradient;
    template <int rank, int dim, unsigned int idx>
    class FEGradient;
    template <int rank, int dim, unsigned int idx>
    class FEHessian;
    template <int rank, int dim, unsigned int idx>
    class FEDiagonalHessian;
    template <int rank, int dim, unsigned int idx>
    class FELaplacian;
    template <typename... Types>
    class SumFEFunctions;
    template <class FEFunctionType>
    class FELiftDivergence;
  }
}

namespace Traits
{
  template <typename Number>
  struct is_block_vector<::dealii::LinearAlgebra::distributed::BlockVector<Number>>
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

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::TestFunction<rank, dim, idx>>
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

  template <int rank, int dim, unsigned int idx>
  struct is_test_function_set<dealii::MatrixFree::TestFunction<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::TestDivergence<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_test_function_set<dealii::MatrixFree::TestDivergence<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::TestCurl<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_test_function_set<dealii::MatrixFree::TestCurl<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::TestSymmetricGradient<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_test_function_set<dealii::MatrixFree::TestSymmetricGradient<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::TestGradient<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_test_function_set<dealii::MatrixFree::TestGradient<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::TestHessian<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_test_function_set<dealii::MatrixFree::TestHessian<rank, dim, idx>>
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

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::FEFunction<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_fe_function_set<dealii::MatrixFree::FEFunction<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::FEDivergence<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_fe_function_set<dealii::MatrixFree::FEDivergence<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::FECurl<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_fe_function_set<dealii::MatrixFree::FECurl<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::FESymmetricGradient<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_fe_function_set<dealii::MatrixFree::FESymmetricGradient<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::FEGradient<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_fe_function_set<dealii::MatrixFree::FEGradient<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::FEHessian<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_fe_function_set<dealii::MatrixFree::FEHessian<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::FEDiagonalHessian<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_fe_function_set<dealii::MatrixFree::FEDiagonalHessian<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_cfl_object<dealii::MatrixFree::FELaplacian<rank, dim, idx>>
  {
    static const bool value = true;
  };

  template <int rank, int dim, unsigned int idx>
  struct is_fe_function_set<dealii::MatrixFree::FELaplacian<rank, dim, idx>>
  {
    static const bool value = true;
  };
}

namespace dealii
{
  namespace MatrixFree
  {
    template <int rank, int dim, unsigned int idx>
    class TestFunction
    {
    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;

      static constexpr unsigned int index = idx;
      static constexpr bool integrate_value = true;
      static constexpr bool integrate_gradient = false;
      static constexpr bool scalar_valued = (TensorTraits::rank > 0);

      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestFunction is vector valued or "
                      "the TestFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestFunction " << index << " " << q << std::endl;
#endif
        phi.template submit_value<index>(value, q);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class TestDivergence
    {
    public:
      static constexpr unsigned int index = idx;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = true;
      typedef Traits::Tensor<rank, dim> TensorTraits;

      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert(FEEvaluation::template rank<index>() > 0,
                      "The proposed FiniteElement has to be "
                      "vector valued for using TestDivergence!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestDivergence " << index << " " << q << std::endl;
#endif
        phi.template submit_divergence<index>(value, q);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class TestSymmetricGradient
    {
    public:
      static constexpr unsigned int index = idx;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = true;
      typedef Traits::Tensor<rank, dim> TensorTraits;

      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestGradient is vector valued or "
                      "the TestGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit SymmetricGradient " << index << " " << q << std::endl;
#endif
        phi.template submit_symmetric_gradient<index>(value, q);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class TestCurl
    {
    public:
      static constexpr unsigned int index = idx;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = true;
      typedef Traits::Tensor<rank, dim> TensorTraits;

      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestCurl is vector valued or "
                      "the TestCurl is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestCurl " << index << " " << q << std::endl;
#endif
        phi.template submit_curl<index>(value, q);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class TestGradient
    {
    public:
      static constexpr unsigned int index = idx;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = true;
      typedef Traits::Tensor<rank, dim> TensorTraits;

      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestGradient is vector valued or "
                      "the TestGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
#ifdef DEBUG_OUTPUT
        std::cout << "submit TestGradient " << index << " " << q << std::endl;
#endif
        phi.template submit_gradient<index>(value, q);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class TestHessian
    {
    public:
      static constexpr unsigned int index = idx;
      static constexpr bool integrate_value = false;
      static constexpr bool integrate_gradient = false;
      typedef Traits::Tensor<rank, dim> TensorTraits;

      template <class FEEvaluation, typename ValueType>
      static void
      submit(FEEvaluation& /*phi*/, unsigned int /*q*/, const ValueType& /*value*/)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 2),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the TestHessian is vector valued or "
                      "the TestHessian is scalar valued and "
                      "the FiniteElement is vector valued!");
        AssertThrow(false, "Not implemented yet!");
      }
    };

    template <int rank, int dim, unsigned int idx>
    TestDivergence<rank - 1, dim, idx>
    div(const TestFunction<rank, dim, idx>&)
    {
      return TestDivergence<rank - 1, dim, idx>();
    }

    template <int rank, int dim, unsigned int idx>
    TestGradient<rank + 1, dim, idx>
    grad(const TestFunction<rank, dim, idx>&)
    {
      return TestGradient<rank + 1, dim, idx>();
    }

    template <int rank, int dim, unsigned int idx>
    TestHessian<rank + 1, dim, idx>
    grad(const TestGradient<rank, dim, idx>&)
    {
      return TestHessian<rank + 1, dim, idx>();
    }

    template <int rank, int dim, unsigned int idx>
    class FEFunction
    {
      const std::string data_name;

    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;
      static constexpr unsigned int index = idx;
      double factor = 1.;

      FEFunction(const std::string& name)
        : data_name(name)
      {
      }

      FEFunction(double new_factor = 1.) { factor = new_factor; }

      auto operator-() const
      {
        const typename std::remove_reference<decltype(*this)>::type newfunction(-factor);
        return newfunction;
      }

      const std::string&
      name() const
      {
        return data_name;
      }

      template <class FEDatas>
      auto value(const FEDatas& phi, unsigned int q) const
      {
        return factor * phi.template get_value<index>(q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 0),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEFunction is vector valued or "
                      "the FEFunction is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<index>(true, false, false);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class FEDivergence
    {
    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;
      static constexpr unsigned int index = idx;
      double factor = 1.;

      explicit FEDivergence(const double new_factor = 1.)
        : factor(new_factor)
      {
      }

      auto operator-() const
      {
        const typename std::remove_reference<decltype(*this)>::type newfunction(-factor);
        return newfunction;
      }

      template <class FEDatas>
      auto value(const FEDatas& phi, unsigned int q) const
      {
        return factor * phi.template get_divergence<index>(q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert(FEEvaluation::template rank<index>() > 0,
                      "The proposed FiniteElement has to be "
                      "vector valued for using FEDivergence!");
        phi.template set_evaluation_flags<index>(false, true, false);
      }
    };

    template <class FEFunctionType>
    class FELiftDivergence
    {
    private:
      const FEFunctionType& fefunction;

    public:
      typedef Traits::Tensor<FEFunctionType::TensorTraits::rank + 2,
                             FEFunctionType::TensorTraits::dim> TensorTraits;
      static constexpr unsigned int index = FEFunctionType::idx;

      FELiftDivergence(const FEFunctionType& fe_function)
        : fefunction(fe_function)
      {
      }

      template <class FEDatas>
      auto value(const FEDatas& phi, unsigned int q) const
      {
        auto value = fefunction.value(phi, q);
        ::dealii::Tensor<TensorTraits::rank, TensorTraits::dim, decltype(value)> lifted_tensor;
        for (unsigned int i = 0; i < TensorTraits::dim; ++i)
          lifted_tensor[i][i] = value;
        return lifted_tensor;
      }

      auto operator-() const { return FELiftDivergence(-fefunction); }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunctionType::set_evaluation_flags(phi);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class FESymmetricGradient
    {
    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;
      static constexpr unsigned int index = idx;
      double factor = 1;

      FESymmetricGradient() = default;

      auto operator-() const
      {
        const typename std::remove_reference<decltype(*this)>::type newfunction(-factor);
        return newfunction;
      }

      template <class FEDatas>
      auto value(const FEDatas& phi, unsigned int q) const
      {
        return factor * phi.template get_symmetric_gradient<index>(q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEGradient is vector valued or "
                      "the FEGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<index>(false, true, false);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class FECurl
    {
    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;
      static constexpr unsigned int index = idx;
      double factor = 1.;

      FECurl(const FEFunction<rank - 1, dim, idx>&) {}

      auto operator-() const
      {
        const typename std::remove_reference<decltype(*this)>::type newfunction(-factor);
        return newfunction;
      }

      template <class FEDatas>
      auto value(const FEDatas& phi, unsigned int q) const
      {
        return factor * phi.template get_curl<index>(q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEGradient is vector valued or "
                      "the FEGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<index>(false, true, false);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class FEGradient
    {
    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;
      static constexpr unsigned int index = idx;
      double factor = 1.;

      FEGradient(const FEFunction<rank - 1, dim, idx>&) {}

      auto operator-() const
      {
        const typename std::remove_reference<decltype(*this)>::type newfunction(-factor);
        return newfunction;
      }

      template <class FEDatas>
      auto value(const FEDatas& phi, unsigned int q) const
      {
        return factor * phi.template get_gradient<index>(q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 1),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEGradient is vector valued or "
                      "the FEGradient is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<index>(false, true, false);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class FELaplacian
    {
    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;
      static constexpr unsigned int index = idx;
      double factor = 1.;

      FELaplacian() = default;

      auto operator-() const
      {
        const typename std::remove_reference<decltype(*this)>::type newfunction(-factor);
        return newfunction;
      }

      template <class FEDatas>
      auto value(const FEDatas& phi, unsigned int q) const
      {
        return factor * phi.template get_laplacian<index>(q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 2),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEHessian is vector valued or "
                      "the FEHessian is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<index>(false, false, true);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class FEDiagonalHessian
    {
    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;
      static constexpr unsigned int index = idx;
      double factor = 1.;

      FEDiagonalHessian() = default;

      auto operator-() const
      {
        const typename std::remove_reference<decltype(*this)>::type newfunction(-factor);
        return newfunction;
      }

      template <class FEDatas>
      auto value(const FEDatas& phi, unsigned int q) const
      {
        return factor * phi.template get_hessian_diagonal<index>(q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 2),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEHessian is vector valued or "
                      "the FEHessian is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<index>(false, false, true);
      }
    };

    template <int rank, int dim, unsigned int idx>
    class FEHessian
    {
      const FEFunction<rank - 2, dim, idx>& base;

    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;
      static constexpr unsigned int index = idx;
      double factor = 1.;

      FEHessian(const FEGradient<rank - 1, dim, idx>&) {}

      auto operator-() const
      {
        const typename std::remove_reference<decltype(*this)>::type newfunction(-factor);
        return newfunction;
      }

      template <class FEDatas>
      auto value(const FEDatas& phi, unsigned int q) const
      {
        return factor * phi.template get_hessian<index>(q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        static_assert((FEEvaluation::template rank<index>() > 0) == (TensorTraits::rank > 2),
                      "Either the proposed FiniteElement is scalar valued "
                      "and the FEHessian is vector valued or "
                      "the FEHessian is scalar valued and "
                      "the FiniteElement is vector valued!");
        phi.template set_evaluation_flags<index>(false, false, true);
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
    div(const FEFunction<rank, dim, idx>& /*f*/)
    {
      return FEDivergence<rank - 1, dim, idx>();
    }

    template <int rank, int dim, unsigned int idx>
    FEHessian<rank + 1, dim, idx>
    grad(const FEGradient<rank, dim, idx>& f)
    {
      return FEHessian<rank + 1, dim, idx>(f);
    }

    template <int rank, int dim, unsigned int idx>
    FELaplacian<rank - 1, dim, idx>
    div(const FEGradient<rank, dim, idx>&)
    {
      return FELaplacian<rank - 1, dim, idx>();
    }

    template <class A, typename Number>
    typename std::enable_if<CFL::Traits::is_fe_function_set<A>::value, A>::type operator*(
      const A& a, const Number factor)
    {
      A tmp = a;
      tmp.factor *= factor;
      return tmp;
    }

    template <typename Number, class A>
    typename std::enable_if<CFL::Traits::is_fe_function_set<A>::value, A>::type operator*(
      const Number& factor, const A& a)
    {
      return a * factor;
    }

    template <typename... Types>
    class SumFEFunctions
    {
    public:
      SumFEFunctions() = delete;
      SumFEFunctions(const SumFEFunctions<Types...>&) = delete;
    };

    template <class FEFunction>
    class SumFEFunctions<FEFunction>
    {
    public:
      typedef Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>
        TensorTraits;

      SumFEFunctions(const FEFunction& summand)
        : summand(summand)
      {
        static_assert(Traits::is_fe_function_set<FEFunction>::value,
                      "You need to construct this with a FEFunction object!");
      }

      template <class NewFEFunction>
      SumFEFunctions<NewFEFunction, FEFunction> operator+(const NewFEFunction& new_summand) const
      {
        static_assert(Traits::is_fe_function_set<NewFEFunction>::value,
                      "Only FEFunction objects can be added!");
        static_assert(TensorTraits::dim == NewFEFunction::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == NewFEFunction::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
        return SumFEFunctions<NewFEFunction, FEFunction>(new_summand, summand);
      }

      template <class NewFEFunction>
      SumFEFunctions<NewFEFunction, FEFunction> operator-(const NewFEFunction& new_summand) const
      {
        return operator+(-new_summand);
      }

      template <class FEEvaluation>
      auto value(FEEvaluation& phi, unsigned int q) const
      {
        return summand.value(phi, q);
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunction::set_evaluation_flags(phi);
      }

      const FEFunction&
      get_summand() const
      {
        return summand;
      }

    private:
      const FEFunction summand;
    };

    template <class FEFunction, typename... Types>
    class SumFEFunctions<FEFunction, Types...> : public SumFEFunctions<Types...>
    {
    public:
      typedef Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>
        TensorTraits;

      template <class FEEvaluation>
      auto value(const FEEvaluation& phi, unsigned int q) const
      {
        const auto own_value = summand.value(phi, q);
        const auto other_value = SumFEFunctions<Types...>::value(phi, q);
        assert_is_compatible(own_value, other_value);
        return own_value + other_value;
      }

      template <class FEEvaluation>
      static void
      set_evaluation_flags(FEEvaluation& phi)
      {
        FEFunction::set_evaluation_flags(phi);
        SumFEFunctions<Types...>::set_evaluation_flags(phi);
      }

      SumFEFunctions(const FEFunction& summand, const Types&... old_sum)
        : SumFEFunctions<Types...>(old_sum...)
        , summand(summand)
      {
        static_assert(Traits::is_fe_function_set<FEFunction>::value,
                      "You need to construct this with a FEFunction object!");
        static_assert(TensorTraits::dim == SumFEFunctions<Types...>::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == SumFEFunctions<Types...>::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
      }

      SumFEFunctions(const FEFunction& summand, const SumFEFunctions<Types...>& old_sum)
        : SumFEFunctions<Types...>(old_sum)
        , summand(summand)
      {
        static_assert(Traits::is_fe_function_set<FEFunction>::value,
                      "You need to construct this with a FEFunction object!");
        static_assert(TensorTraits::dim == SumFEFunctions<Types...>::TensorTraits::dim,
                      "You can only add tensors of equal dimension!");
        static_assert(TensorTraits::rank == SumFEFunctions<Types...>::TensorTraits::rank,
                      "You can only add tensors of equal rank!");
      }

      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              SumFEFunctions<NewFEFunction, FEFunction, Types...>>::type
      operator+(const NewFEFunction& new_summand) const
      {
        return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_summand, *this);
      }

      template <class NewFEFunction, typename... NewTypes>
      typename std::enable_if<
        CFL::Traits::is_fe_function_set<NewFEFunction>::value,
        SumFEFunctions<NewTypes..., NewFEFunction, FEFunction, Types...>>::type
      operator+(const SumFEFunctions<NewFEFunction, NewTypes...>& new_sum) const
      {
        return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_sum.get_summand(), *this) +
               SumFEFunctions<NewTypes...>(
                 static_cast<const SumFEFunctions<NewTypes...>&>(new_sum));
      }

      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              SumFEFunctions<NewFEFunction, FEFunction, Types...>>::type
      operator+(const SumFEFunctions<NewFEFunction>& new_sum) const
      {
        return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_sum.get_summand(), *this);
      }

      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              SumFEFunctions<NewFEFunction, FEFunction, Types...>>::type
      operator-(const NewFEFunction& new_summand) const
      {
        return operator+(-new_summand);
      }

      template <class NewFEFunction, typename... NewTypes>
      typename std::enable_if<
        CFL::Traits::is_fe_function_set<NewFEFunction>::value,
        SumFEFunctions<NewTypes..., NewFEFunction, FEFunction, Types...>>::type
      operator-(const SumFEFunctions<NewFEFunction, NewTypes...>& new_sum) const
      {
        return operator+(-new_sum);
      }

      template <class NewFEFunction>
      typename std::enable_if<CFL::Traits::is_fe_function_set<NewFEFunction>::value,
                              SumFEFunctions<NewFEFunction, FEFunction, Types...>>::type
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
      const FEFunction summand;
    };

    template <class FEFunction1, class FEFunction2>
    typename std::enable_if<Traits::is_fe_function_set<FEFunction1>::value &&
                              Traits::is_fe_function_set<FEFunction2>::value,
                            SumFEFunctions<FEFunction2, FEFunction1>>::type
    operator+(const FEFunction1& old_fe_function, const FEFunction2& new_fe_function)
    {

      static_assert(FEFunction1::TensorTraits::dim == FEFunction2::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(FEFunction1::TensorTraits::rank == FEFunction2::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
      return SumFEFunctions<FEFunction2, FEFunction1>(new_fe_function, old_fe_function);
    }

    template <class FEFunction, typename... Types>
    typename std::enable_if<Traits::is_fe_function_set<FEFunction>::value,
                            SumFEFunctions<FEFunction, Types...>>::type
    operator+(const FEFunction& new_fe_function, const SumFEFunctions<Types...>& old_fe_function)
    {
      return old_fe_function + new_fe_function;
    }

    template <class FEFunction1, class FEFunction2>
    typename std::enable_if<Traits::is_fe_function_set<FEFunction1>::value &&
                              Traits::is_fe_function_set<FEFunction2>::value,
                            SumFEFunctions<FEFunction2, FEFunction1>>::type
    operator-(const FEFunction1& old_fe_function, const FEFunction2& new_fe_function)
    {
      return old_fe_function + (-new_fe_function);
    }

    template <class FEFunction, typename... Types>
    typename std::enable_if<Traits::is_fe_function_set<FEFunction>::value,
                            SumFEFunctions<FEFunction, Types...>>::type
    operator-(const FEFunction& new_fe_function, const SumFEFunctions<Types...>& old_fe_function)
    {
      return -(old_fe_function - new_fe_function);
    }
  }
}
}

#endif // CFL_DEALII_MATRIXFREE_H
