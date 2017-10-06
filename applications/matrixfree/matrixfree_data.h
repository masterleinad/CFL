#ifndef MATRIXFREE_DATA_H
#define MATRIXFREE_DATA_H

#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <cfl/base/fefunctions.h>

#include <cfl/matrixfree/fe_data.h>
#include <cfl/matrixfree/matrix_free_integrator.h>

#include <utility>

/**
* @brief Class which wraps the operational aspects of MatrixFree implementation
*
* This class provides a simplified interface to the integration kernel using
* MatrixFree evaluation technique as prescribed in step-37
*
*
*/
template <int dim, class FEDatas, class Forms, typename VectorType>
class MatrixFreeData
{
  const dealii::MappingQ<dim, dim> mapping;
  dealii::SphericalManifold<dim> sphere;
  dealii::Triangulation<dim> tr;
  std::vector<std::unique_ptr<dealii::DoFHandler<dim>>> dh_ptr_vector;
  std::vector<const dealii::DoFHandler<dim>*> dh_const_ptr_vector;
  std::vector<std::unique_ptr<dealii::ConstraintMatrix>> constraint_ptr_vector;
  std::vector<const dealii::ConstraintMatrix*> constraint_const_ptr_vector;
  std::vector<dealii::Quadrature<1>> quadrature_vector;
  std::shared_ptr<dealii::MatrixFree<dim, double>> mf;
  std::shared_ptr<FEDatas> fe_datas;
  std::shared_ptr<Forms> forms;
  const dealii::Quadrature<1> quadrature;
  MatrixFreeIntegrator<dim, VectorType, Forms, FEDatas> integrator;

public:
  // constructor for multiple FiniteElements
  MatrixFreeData(unsigned int grid_index, unsigned int refine,
                 const std::vector<dealii::FiniteElement<dim>*>& fe,
                 std::shared_ptr<FEDatas> fe_datas_, std::shared_ptr<Forms> forms_)
    : mapping(FEDatas::max_degree)
    , fe_datas(std::move(fe_datas_))
    , forms(std::move(forms_))
    , quadrature(FEDatas::max_degree + 1)
  {
    AssertThrow(!fe.empty(), dealii::ExcInternalError());
    if (grid_index == 0)
      dealii::GridGenerator::hyper_cube(tr);
    else if (grid_index == 1)
    {
      dealii::GridGenerator::hyper_ball(tr);
      tr.set_manifold(0, sphere);
      tr.set_all_manifold_ids(0);
    }
    else
      throw std::logic_error(std::string("Unknown grid index") + std::to_string(grid_index));
    tr.refine_global(refine);

    for (size_t i = 0; i < fe.size(); ++i)
    {
      dh_ptr_vector.push_back(std::make_unique<dealii::DoFHandler<dim>>());
      dh_ptr_vector[i]->initialize(tr, *(fe[i]));
      dh_const_ptr_vector.push_back(dh_ptr_vector[i].get());
      // dh_vector[i].initialize_local_block_info();
      constraint_ptr_vector.push_back(std::make_unique<dealii::ConstraintMatrix>());
      constraint_const_ptr_vector.push_back(constraint_ptr_vector[i].get());
      quadrature_vector.push_back(dealii::QGauss<1>(fe[i]->degree + 1));
    }

    dealii::deallog << "Grid type " << grid_index << " Cells " << tr.n_active_cells() << " DoFs ";
    for (size_t i = 0; i < fe.size() - 1; ++i)
      dealii::deallog << dh_ptr_vector[i]->n_dofs() << "+";
    dealii::deallog << dh_ptr_vector[fe.size() - 1]->n_dofs() << std::endl;

    typename dealii::MatrixFree<dim, double>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = dealii::MatrixFree<dim, double>::AdditionalData::none;
    addit_data.tasks_block_size = 3;
    addit_data.level_mg_handler = dealii::numbers::invalid_unsigned_int;
    addit_data.build_face_info = true;

    mf = std::make_shared<dealii::MatrixFree<dim, double>>();
    mf->reinit(
      mapping, dh_const_ptr_vector, constraint_const_ptr_vector, quadrature_vector, addit_data);

    integrator.initialize(mf, forms, fe_datas);
  }

  // constructor for multiple FiniteElements
  MatrixFreeData(unsigned int grid_index, unsigned int refine,
                 const std::vector<dealii::FiniteElement<dim>*>& fe, FEDatas fe_datas_,
                 Forms forms_)
    : MatrixFreeData(grid_index, refine, fe, std::make_shared<FEDatas>(fe_datas_),
                     std::make_shared<Forms>(forms_))
  {
  }

  void
  resize_vector(VectorType& v) const
  {
    if
      constexpr(CFL::Traits::is_block_vector<VectorType>::value)
      {
        AssertDimension(v.n_blocks(), dh_ptr_vector.size());
        for (unsigned int i = 0; static_cast<size_t>(i) < v.n_blocks(); ++i)
        {
          mf->initialize_dof_vector(v.block(i), i);
          if
            constexpr(
              std::is_same<
                dealii::LinearAlgebra::distributed::BlockVector<typename VectorType::value_type>,
                VectorType>::value) mf->initialize_dof_vector(v.block(i), i);
          else
            v.block(i).reinit(dh_ptr_vector[i].n_dofs());
          std::cout << "Vector " << i << " has size " << v.block(i).size() << std::endl;
        }
      }
    else
    {
      AssertDimension(dh_ptr_vector.size(), 1);
      if
        constexpr(
          std::is_same<dealii::LinearAlgebra::distributed::Vector<typename VectorType::value_type>,
                       VectorType>::value) mf->initialize_dof_vector(v, 0);
      else
        v.reinit(dh_ptr_vector[0].n_dofs());
      std::cout << "Vector has size " << v.size() << std::endl;
    }
  }

  void
  vmult(VectorType& dst, const VectorType& src) const
  {
    integrator.vmult(dst, src);
  }

  void
  vmult_add(VectorType& dst, const VectorType& src) const
  {
    integrator.vmult_add(dst, src);
  }
};
#endif // MATRIXFREE_DATA_H
