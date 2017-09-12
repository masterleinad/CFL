#ifndef cfl_dealii_h
#define cfl_dealii_h

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

    template <int dim>
    class ScalarTestFunction;
    template <int dim>
    class ScalarTestGradient;
    template <int dim>
    class ScalarTestHessian;

    template <int rank, int dim>
    class FEFunction;
    template <int rank, int dim>
    class FEGradient;
    template <int rank, int dim>
    class FEHessian;
  }
}

namespace Traits
{
  template <int dim>
  struct test_function_set_type<dealii::MeshWorker::ScalarTestFunction<dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int dim>
  struct test_function_set_type<dealii::MeshWorker::ScalarTestGradient<dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int dim>
  struct test_function_set_type<dealii::MeshWorker::ScalarTestHessian<dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int rank, int dim>
  struct fe_function_set_type<dealii::MeshWorker::FEFunction<rank, dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int rank, int dim>
  struct fe_function_set_type<dealii::MeshWorker::FEGradient<rank, dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };

  template <int rank, int dim>
  struct fe_function_set_type<dealii::MeshWorker::FEHessian<rank, dim>>
  {
    static const ObjectType value = ObjectType::cell;
  };
}

namespace dealii
{
  namespace MeshWorker
  {
    template <int dim>
    class ScalarTestFunction
    {
      /// The constant index data used in local integration
      TestFunctionIdentifier id;

      /// Pointer to the DoFInfo object storing the result
      mutable ::dealii::MeshWorker::DoFInfo<dim, dim> const* di;
      /// Pointer to the IntegrationInfo object containing the local data
      mutable ::dealii::MeshWorker::IntegrationInfo<dim, dim> const* ii;

      friend class ScalarTestGradient<dim>;
      friend class ScalarTestHessian<dim>;

    public:
      typedef Traits::Tensor<0, dim> TensorTraits;

      constexpr ScalarTestFunction(unsigned int fe_index, unsigned int block_index)
        : id{ fe_index, block_index }
        , di{ nullptr }
        , ii{ nullptr }
      {
      }

      void
      reinit(const ::dealii::MeshWorker::DoFInfo<dim, dim>& di,
             const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii) const
      {
        (this)->di = &di;
        (this)->ii = &ii;
      }

      double
      evaluate(unsigned int quadrature_index, unsigned int test_function_index) const
      {
        Assert(ii != nullptr, ::dealii::ExcInternalError());
        return ii->fe_values(id.fe_index).shape_value(test_function_index, quadrature_index);
      }
    };

    template <int dim>
    class ScalarTestGradient
    {
      const ScalarTestFunction<dim>& base;
      friend class ScalarTestHessian<dim>;

    public:
      typedef Traits::Tensor<1, dim> TensorTraits;

      ScalarTestGradient(const ScalarTestFunction<dim>& base)
        : base(base)
      {
      }

      void
      reinit(const ::dealii::MeshWorker::DoFInfo<dim, dim>& di,
             const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii) const
      {
        base.reinit(di, ii);
      }

      double
      evaluate(unsigned int quadrature_index, unsigned int test_function_index, int comp) const
      {
        Assert(base.ii != nullptr, ::dealii::ExcInternalError());
        return base.ii->fe_values(base.id.fe_index)
          .shape_grad(test_function_index, quadrature_index)[comp];
      }
    };

    template <int dim>
    class ScalarTestHessian
    {
      const ScalarTestFunction<dim>& base;

    public:
      typedef Traits::Tensor<2, dim> TensorTraits;

      ScalarTestHessian(const ScalarTestGradient<dim>& grad)
        : base(grad.base)
      {
      }

      void
      reinit(const ::dealii::MeshWorker::DoFInfo<dim, dim>& di,
             const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii) const
      {
        base.reinit(di, ii);
      }

      double
      evaluate(unsigned int quadrature_index, unsigned int test_function_index, int comp1,
               int comp2) const
      {
        Assert(base.ii != nullptr, ::dealii::ExcInternalError());
        return base.ii->fe_values(base.id.fe_index)
          .shape_hessian(test_function_index, quadrature_index)(comp1, comp2);
      }
    };

    template <int dim>
    ScalarTestGradient<dim>
    grad(const ScalarTestFunction<dim>& func)
    {
      return ScalarTestGradient<dim>(func);
    }

    template <int dim>
    ScalarTestHessian<dim>
    grad(const ScalarTestGradient<dim>& func)
    {
      return ScalarTestHessian<dim>(func);
    }

    template <int rank, int dim>
    class FEFunction
    {
      const std::string data_name;
      const unsigned int first_component;
      mutable unsigned int data_index;
      mutable ::dealii::MeshWorker::IntegrationInfo<dim, dim> const* info;

      friend class FEGradient<rank, dim>;
      friend class FEHessian<rank, dim>;

    public:
      typedef Traits::Tensor<rank, dim> TensorTraits;

      FEFunction(const std::string& name, const unsigned int first)
        : data_name(name)
        , first_component(first)
        , data_index(::dealii::numbers::invalid_unsigned_int)
        , info(nullptr)
      {
      }

      const std::string&
      name() const
      {
        return data_name;
      }

      void
      anchor(const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
             const ::dealii::MeshWorker::LocalIntegrator<dim>& li) const
      {
        info = &ii;

        // short cut
        if (data_index < li.input_vector_names.size() &&
            data_name == li.input_vector_names[data_index])
          return;

        unsigned int i = 0;

        while (i < li.input_vector_names.size())
        {
          if (data_name == li.input_vector_names[i])
          {
            data_index = i;
            break;
          }
          ++i;
        }
        if (i == li.input_vector_names.size())
          throw std::invalid_argument(std::string("Vector name not found: ") + data_name);
      }

      // TODO: only implemented for scalars yet
      double
      evaluate(unsigned int quadrature_index) const
      {
        Assert(info != nullptr && data_index != ::dealii::numbers::invalid_unsigned_int,
               ::dealii::ExcInternalError());
        return info->values[data_index][first_component][quadrature_index];
      }
    };

    template <int rank, int dim>
    class FEGradient
    {
      const FEFunction<rank, dim>& base;
      friend class FEHessian<rank, dim>;

    public:
      typedef Traits::Tensor<rank + 1, dim> TensorTraits;

      FEGradient(const FEFunction<rank, dim>& base)
        : base(base)
      {
      }

      void
      anchor(const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
             const ::dealii::MeshWorker::LocalIntegrator<dim>& li) const
      {
        base.anchor(ii, li);
      }

      // TODO: only implemented for scalars yet
      double
      evaluate(unsigned int quadrature_index, unsigned int comp) const
      {
        Assert(base.info != nullptr, ::dealii::ExcInternalError());

        AssertIndexRange(base.data_index, base.info->gradients.size());

        AssertIndexRange(base.first_component, base.info->gradients[base.data_index].size());
        AssertIndexRange(quadrature_index,
                         base.info->gradients[base.data_index][base.first_component].size());
        AssertIndexRange(comp, dim);
        return base.info->gradients[base.data_index][base.first_component][quadrature_index][comp];
      }
    };

    template <int rank, int dim>
    class FEHessian
    {
      const FEFunction<rank, dim>& base;

    public:
      typedef Traits::Tensor<rank + 2, dim> TensorTraits;

      FEHessian(const FEGradient<rank, dim>& grad)
        : base(grad.base)
      {
      }

      void
      anchor(const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
             const ::dealii::MeshWorker::LocalIntegrator<dim>& li) const
      {
        base.anchor(ii, li);
      }

      // TODO: only implemented for scalars yet
      double
      evaluate(unsigned int quadrature_index, unsigned int comp1, unsigned int comp2) const
      {
        Assert(base.info != nullptr, ::dealii::ExcInternalError());
        return base.info
          ->hessians[base.data_index][base.first_component][quadrature_index][comp1][comp2];
      }
    };

    template <int rank, int dim>
    FEGradient<rank, dim>
    grad(const FEFunction<rank, dim>& f)
    {
      return FEGradient<rank, dim>(f);
    }

    template <int rank, int dim>
    FEHessian<rank, dim>
    grad(const FEGradient<rank, dim>& f)
    {
      return FEHessian<rank, dim>(f);
    }

    template <class TEST, class EXPR, FormKind kind_of_form>
    void
    anchor(const Form<TEST, EXPR, kind_of_form>& form,
           const ::dealii::MeshWorker::IntegrationInfo<TEST::TensorTraits::dim,
                                                       TEST::TensorTraits::dim>& ii,
           const ::dealii::MeshWorker::LocalIntegrator<TEST::TensorTraits::dim>& li)
    {
      form.expr.anchor(ii, li);
    }

    template <class TEST, class EXPR, FormKind kind_of_form>
    void
    reinit(
      const Form<TEST, EXPR, kind_of_form>& form,
      const ::dealii::MeshWorker::DoFInfo<TEST::TensorTraits::dim, TEST::TensorTraits::dim>& di,
      const ::dealii::MeshWorker::IntegrationInfo<TEST::TensorTraits::dim, TEST::TensorTraits::dim>&
        ii)
    {
      form.test.reinit(di, ii);
    }

    template <class T, int dim = T::TensorTraits::dim>
    std::enable_if<Traits::needs_anchor<T>::type>
    anchor(const T& t, const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
           const ::dealii::MeshWorker::LocalIntegrator<dim>& li)
    {
      t.anchor(ii, li);
    }

    template <class T, int dim = T::TensorTraits::dim>
    std::enable_if<Traits::is_unary_operator<T>::type>
    anchor(const T& t, const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
           const ::dealii::MeshWorker::LocalIntegrator<dim>& li)
    {
      anchor(t.base, ii, li);
    }
  }
}
}

#endif
