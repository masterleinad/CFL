#ifndef cfl_dealii_meshworker_h
#define cfl_dealii_meshworker_h

#include <cfl/forms.h>
#include <cfl/traits.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/output.h>
#include <deal.II/meshworker/simple.h>

// This is an ugly workaround to be able to use AssertIndexRange
// because we are always in the wrong namespace.
#undef AssertIndexRange
#define AssertIndexRange(index, range)                                                             \
  Assert((index) < (range), ::dealii::ExcIndexRange((index), 0, (range)))

using namespace ::dealii::MeshWorker;

namespace CFL
{
/**
 * \brief Interface to the deal.II library
 */
namespace dealii
{
  /**
   * \brief The terminal objects based on dealii::Meshworker classes.
   */
  namespace MeshWorker
  {
    /**
     * In MeshWorker, a test function space is characterized by the
     * FEValues object it accesses and by the block in the result
     * vector into which the data is written.
     *
     * This data is independent of whether we use the function itself
     * or its derivatives. It is also independent of whether the space
     * is scalar or vector valued.
     *
     * The class is just a simple wrapper, such that a struct seemed
     * reasonable.
     */
    struct TestFunctionIdentifier
    {
      /// The number of the FEValues object in IntegrationInfo
      unsigned int fe_index;
      /// The block in the finite element system associated with these test functions
      unsigned int block_index;
    };

    template <int order, int dim>
    class TestFunction;
    template <int order, int dim>
    class TestGradient;
    template <int order, int dim>
    class TestHessian;

    template <int order, int dim>
    class FEFunction;
    template <int order, int dim>
    class FEGradient;
    template <int order, int dim>
    class FEHessian;
  }
}

namespace Traits
{
  template <int order, int dim>
  struct test_function_set_type<dealii::MeshWorker::TestFunction<order,dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int order, int dim>
  struct test_function_set_type<dealii::MeshWorker::TestGradient<order,dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int order, int dim>
  struct test_function_set_type<dealii::MeshWorker::TestHessian<order,dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int order, int dim>
  struct fe_function_set_type<dealii::MeshWorker::FEFunction<order, dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int order, int dim>
  struct fe_function_set_type<dealii::MeshWorker::FEGradient<order, dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int order, int dim>
  struct fe_function_set_type<dealii::MeshWorker::FEHessian<order, dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };
}

namespace dealii
{
  namespace MeshWorker
  {
    template <int order, int dim>
    class TestFunction
    {
      /// The constant index data used in local integration
      const TestFunctionIdentifier ident;

      friend class TestGradient<order, dim>;
      friend class TestHessian<order, dim>;

    public:
      /// Remove after reducing Forms
      static const bool integration_flags = false;
      typedef Traits::Tensor<order, dim> TensorTraits;

      constexpr TestFunction(unsigned int fe_index, unsigned int block_index)
        : ident{ fe_index, block_index }
      {
      }

      constexpr TestFunctionIdentifier id() const
      {
	return ident;
      }
      
      double
      value(unsigned int test_index,
	    const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
	    unsigned int quadrature_index) const
      {
	static_assert(order==0, "Tensor test function used without tensor coordinate");
        return ii.fe_values(id().fe_index).shape_value(test_index, quadrature_index);
      }
      
      double
      value(unsigned int d, unsigned int test_index,
	    const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
	    unsigned int quadrature_index) const
      {
	static_assert(order==1, "Tensor order and number of tensor coordinates do not match");
        return ii.fe_values(id().fe_index).shape_value_component(test_index, quadrature_index, d);
      }
    };

    template <int order, int dim>
    class TestGradient
    {
      const TestFunction<order, dim>& base;
      friend class TestHessian<order, dim>;

    public:
      /// Remove after reducing Forms
      static const bool integration_flags = false;
      typedef Traits::Tensor<order+1, dim> TensorTraits;

      TestGradient(const TestFunction<order,dim>& base)
        : base {base}
      {
      }

      constexpr TestFunctionIdentifier id() const
      {
	return base.ident;
      }
      
      double
      value(int d,
	    unsigned int test_index,
	    const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
	    unsigned int quadrature_index) const
      {
	static_assert(order==0, "Tensor order and number of tensor coordinates do not match");
        return ii.fe_values(id().fe_index).shape_grad(test_index, quadrature_index)[d];
      }

      double
      value(int d1, int d2,
	    unsigned int test_index,
	    const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
	    unsigned int quadrature_index) const
      {
	static_assert(order==1, "Tensor order and number of tensor coordinates do not match");
        return ii.fe_values(id().fe_index).shape_grad_component(test_index, quadrature_index,d2)[d1];
      }
};

    template <int order, int dim>
    class TestHessian
    {
      const TestFunction<order, dim>& base;

    public:
      /// Remove after reducing Forms
      static const bool integration_flags = false;
      typedef Traits::Tensor<order+2, dim> TensorTraits;

      TestHessian(const TestFunction<order,dim>& base)
        : base {base}
      {
      }

      TestHessian(const TestGradient<order,dim>& grad)
        : base {grad.base}
      {
      }

      constexpr TestFunctionIdentifier id() const
      {
	return base.ident;
      }
      
      double
      value(int d1, int d2,
	    unsigned int test_index,
	    const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
	    unsigned int quadrature_index) const
      {
	static_assert(order==0, "Tensor order and number of tensor coordinates do not match");
        return ii.fe_values(id().fe_index).shape_hessian(test_index, quadrature_index)[d1][d2];
      }
    };

    template <int order, int dim>
    TestGradient<order,dim>
    grad(const TestFunction<order, dim>& func)
    {
      return TestGradient<order, dim>(func);
    }

    template <int order, int dim>
    TestHessian<order,dim>
    grad(const TestGradient<order, dim>& func)
    {
      return TestHessian<order, dim>(func);
    }

    template <int order, int dim>
    class FEFunction
    {
      const unsigned int data_index;
      const unsigned int first_component;

      friend class FEGradient<order, dim>;
      friend class FEHessian<order, dim>;

    public:
      typedef Traits::Tensor<order, dim> TensorTraits;

      FEFunction(const unsigned int data_index, const unsigned int first)
        : data_index(data_index)
        , first_component(first)
      {
      }

      double value(const IntegrationInfo<dim, dim>& info,
		   unsigned int quadrature_index) const
      {
	static_assert(order==0, "Scalar used with tensor coordinate");
        Assert(data_index != ::dealii::numbers::invalid_unsigned_int,
               ::dealii::ExcInternalError());
        return info.values[data_index][first_component][quadrature_index];
      }


      double value(unsigned int d,
		   const IntegrationInfo<dim, dim>& info,
		   unsigned int quadrature_index) const
      {
	static_assert(order==1, "Wrong number of tensor coordinates");
        Assert(data_index != ::dealii::numbers::invalid_unsigned_int,
               ::dealii::ExcInternalError());
        return info.values[data_index][first_component+d][quadrature_index];
      }
    };

    /**
     * The gradient of a meshworker defined finite element function.
     *
     * The gradient is of one tensor order higher than the function
     * itself, adding an index to the front of the set of tensor
     * coordinates.
     */
    template <int order, int dim>
    class FEGradient
    {
      const FEFunction<order, dim>& base;
      friend class FEHessian<order, dim>;

    public:
      typedef Traits::Tensor<order + 1, dim> TensorTraits;

      constexpr FEGradient(const FEFunction<order, dim>& base)
        : base(base)
      {
      }

      double value(unsigned int d,
		   const IntegrationInfo<dim, dim>& info,
		   unsigned int quadrature_index) const
      {
	static_assert(order==0, "Wrong number of tensor coordinates");

        AssertIndexRange(base.data_index, info.gradients.size());

        AssertIndexRange(base.first_component, info.gradients[base.data_index].size());
        AssertIndexRange(quadrature_index,
                         info.gradients[base.data_index][base.first_component].size());
        AssertIndexRange(d, dim);
        return info.gradients[base.data_index][base.first_component][quadrature_index][d];
      }


      double value(unsigned int d1, unsigned int d2,
		   const IntegrationInfo<dim, dim>& info,
		   unsigned int quadrature_index) const
      {
	static_assert(order==1, "Wrong number of tensor coordinates");
        AssertIndexRange(base.data_index, info.gradients.size());

        AssertIndexRange(base.first_component, info.gradients[base.data_index].size());
        AssertIndexRange(quadrature_index,
                         info.gradients[base.data_index][base.first_component].size());
        AssertIndexRange(d1, dim);
        return info.gradients[base.data_index][base.first_component+d2][quadrature_index][d1];
      }
    };

    template <int order, int dim>
    class FEHessian
    {
      const FEFunction<order, dim>& base;

    public:
      typedef Traits::Tensor<order + 2, dim> TensorTraits;

      constexpr FEHessian(const FEGradient<order, dim>& grad)
        : base(grad.base)
      {
      }

      constexpr FEHessian(const FEFunction<order, dim>& base)
        : base(base)
      {
      }

      typename std::enable_if<order==0,double>::type
      value(unsigned int d1, unsigned int d2,
	    const IntegrationInfo<dim, dim>& info,
	    unsigned int quadrature_index) const
      {
        return info.hessians[base.data_index][base.first_component][quadrature_index][d1][d2];
      }
    };

    template <int order, int dim>
    FEGradient<order, dim>
    grad(const FEFunction<order, dim>& f)
    {
      return FEGradient<order, dim>(f);
    }
    
    template <int order, int dim>
    FEHessian<order, dim>
    grad(const FEGradient<order, dim>& f)
    {
      return FEHessian<order, dim>(f);
    }
    
  }
}
}

#endif
