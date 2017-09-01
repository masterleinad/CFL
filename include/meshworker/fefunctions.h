#ifndef MESHWORKER_FEFUNCTIONS_H
#define MESHWORKER_FEFUNCTIONS_H

#include <cfl/traits.h>

/**
 * \brief The terminal objects based on dealii::Meshworker classes.
 */
namespace CFL
{
namespace dealiiMeshWorker
{
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

    namespace Traits
    {
      template <int dim>
      struct test_function_set_type<dealiiMeshWorker::ScalarTestFunction<dim>>
      {
        static const ObjectType value = ObjectType::cell;
      };

      template <int dim>
      struct test_function_set_type<dealiiMeshWorker::ScalarTestGradient<dim>>
      {
        static const ObjectType value = ObjectType::cell;
      };

      template <int dim>
      struct test_function_set_type<dealiiMeshWorker::ScalarTestHessian<dim>>
      {
        static const ObjectType value = ObjectType::cell;
      };

      template <int rank, int dim>
      struct fe_function_set_type<dealiiMeshWorker::FEFunction<rank, dim>>
      {
        static const ObjectType value = ObjectType::cell;
      };

      template <int rank, int dim>
      struct fe_function_set_type<dealiiMeshWorker::FEGradient<rank, dim>>
      {
        static const ObjectType value = ObjectType::cell;
      };

      template <int rank, int dim>
      struct fe_function_set_type<dealiiMeshWorker::FEHessian<rank, dim>>
      {
        static const ObjectType value = ObjectType::cell;
      };
    }

namespace dealiiMeshWorker
{
    template <int dim>
    class ScalarTestFunction
    {
      /// Index of the dealii::FEValues object in IntegrationInfo
      unsigned int index;

      mutable ::dealii::MeshWorker::IntegrationInfo<dim, dim> const* ii;

      friend class ScalarTestGradient<dim>;
      friend class ScalarTestHessian<dim>;

    public:
      typedef Traits::Tensor<0, dim> TensorTraits;

      ScalarTestFunction(unsigned int index)
        : index(index)
        , ii(nullptr)
      {
      }

      void
      reinit(const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii) const
      {
        (this)->ii = &ii;
      }

      double
      evaluate(unsigned int quadrature_index, unsigned int test_function_index) const
      {
        Assert(ii != nullptr, ::dealii::ExcInternalError());
        return ii->fe_values(index).shape_value(test_function_index, quadrature_index);
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
      reinit(const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii) const
      {
        base.reinit(ii);
      }

      double
      evaluate(unsigned int quadrature_index, unsigned int test_function_index, int comp) const
      {
        Assert(base.ii != nullptr, ::dealii::ExcInternalError());
        return base.ii->fe_values(base.index)
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
      reinit(const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii) const
      {
        base.reinit(ii);
      }

      double
      evaluate(unsigned int quadrature_index, unsigned int test_function_index, int comp1,
               int comp2) const
      {
        Assert(base.ii != nullptr, ::dealii::ExcInternalError());
        return base.ii->fe_values(base.index)
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
}
}

#endif // FEFUNCTIONS_H
