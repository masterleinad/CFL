#ifndef cfl_dealii_h
#define cfl_dealii_h

#include <cfl/traits.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>

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
    template <int dim> class ScalarTestFunction;
    template <int dim> class ScalarTestGradient;
    template <int dim> class ScalarTestHessian;

    template <int rank, int dim> class FEFunction;
    template <int rank, int dim> class FEGradient;
    template <int rank, int dim> class FEHessian;
    
    template <int dim>
    class ScalarTestFunction
    {
	/// Index of the dealii::FEValues object in IntegrationInfo
	unsigned int index;

	const ::dealii::FEValuesBase<dim,dim>* fe;

	friend class ScalarTestGradient<dim>;
	friend class ScalarTestHessian<dim>;
	
      public:
	typedef Traits::Tensor<0, dim> Traits;
	Traits traits;

	ScalarTestFunction(unsigned int index)
			:
			index(index), fe(0)
	  {}

	void bind (const ::dealii::MeshWorker::IntegrationInfo<dim,dim>& ii)
	  {
	    fe = &ii.fe_values(index);
	  }

	double evaluate (unsigned int quadrature_index, unsigned int test_function_index) const
	  {
	    return fe->shape_function_value(test_function_index, quadrature_index);
	  }	
    };
    
    template <int dim>
    class ScalarTestGradient
    {
	const ScalarTestFunction<dim>& base;
	friend class ScalarTestHessian<dim>;
	
      public:
	typedef Traits::Tensor<1, dim> Traits;
	Traits traits;

	ScalarTestGradient (const ScalarTestFunction<dim>& base)
			:
			base(base)
	  {}
	
	double evaluate (unsigned int quadrature_index, unsigned int test_function_index,
			 int comp) const
	  {
	    return base->fe->shape_grad(test_function_index, quadrature_index)(comp);
	  }		
    };
    
    
    template <int dim>
    class ScalarTestHessian
    {
	const ScalarTestFunction<dim>& base;
	
      public:
	typedef Traits::Tensor<2, dim> Traits;
	Traits traits;

	ScalarTestHessian (const ScalarTestGradient<dim>& grad)
			:
			base(grad.base)
	  {}
	
	double evaluate (unsigned int quadrature_index, unsigned int test_function_index,
			 int comp1, int comp2) const
	  {
	    return base->fe->shape_hessian(test_function_index, quadrature_index)(comp1,comp2);
	  }
    };

    template <int dim>
    ScalarTestGradient<dim> grad(const ScalarTestFunction<dim>& func)
    {
      return ScalarTestGradient<dim>(func);
    }

    template <int dim>
    ScalarTestHessian<dim> grad(const ScalarTestGradient<dim>& func)
    {
      return ScalarTestHessian<dim>(func);
    }


    template <int rank, int dim>
    class FEFunction
    {
      const std::string data_name;
	const unsigned int first_component;
	unsigned int data_index;
	const ::dealii::MeshWorker::LocalIntegrator<dim>& info;
	
	friend class FEGradient<rank, dim>;
	friend class FEHessian<rank, dim>;
	
    public:
      typedef Traits::Tensor<rank, dim> Traits;
      FEFunction(const std::string& name, const unsigned int first)
	: data_name(name), first_component(first)
      {}

      const std::string& name() const
      {
	return data_name;
      }
      
	void bind (const ::dealii::MeshWorker::IntegrationInfo<dim,dim>& ii,
		   const ::dealii::MeshWorker::LocalIntegrator<dim>& li)
      {
	unsigned int i = 0;
	
	while (i<li.input_vector_names.size())
	  {
	    if (data_name == li.input_vector_names[i])
	      {
		data_index = i;
		break;
	      }
	    ++i
	  }
	if (i==li.input_vector_names.size())
	  throw std::invalid_argument(std::string("Vector name not found: ") + data_name);
      }
	//TODO: only implemented for scalars yet
	double evaluate (unsigned int quadrature_index)
	  {}
	
    };

    template <int rank, int dim>
    class FEGradient
    {
	const FEFunction<rank,dim>& base;
      public:
      typedef Traits::Tensor<rank+1, dim> Traits;
	FEGradient(const FEFunction<rank,dim>& base)
			:
			base(base)
	  {}
	
    };
    
    
  }
}
  
}


#endif










