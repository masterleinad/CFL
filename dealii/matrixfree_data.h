#ifndef MATRIXFREE_DATA_H
#define MATRIXFREE_DATA_H

#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <cfl/dealii_matrixfree.h>

#include <dealii/fe_data.h>
#include <dealii/matrix_free_integrator.h>

#include <utility>

template <int dim, class FEDatas>
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
  dealii::MatrixFree<dim, double> mf;
  std::shared_ptr<FEDatas> fe_datas;
  const dealii::Quadrature<1> quadrature;

public:
  // constructor for multiple FiniteElements
  MatrixFreeData(unsigned int grid_index, unsigned int refine,
                 const std::vector<dealii::FiniteElement<dim>*>& fe,
                 std::shared_ptr<FEDatas> fe_datas_)
    : mapping(FEDatas::max_degree)
    , fe_datas(std::move(fe_datas_))
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
  }

  // constructor for multiple FiniteElements
  MatrixFreeData(unsigned int grid_index, unsigned int refine,
                 const std::vector<dealii::FiniteElement<dim>*>& fe, FEDatas fe_datas_)
    : MatrixFreeData(grid_index, refine, fe, std::make_shared<FEDatas>(fe_datas_))
  {
  }

  void
  initialize()
  {
    mf.reinit(mapping, dh_const_ptr_vector, constraint_const_ptr_vector, quadrature_vector);
    fe_datas->initialize(mf);
  }

  void
  resize_vector(dealii::Vector<double>& v) const
  {
    AssertDimension(v.size(), dh_ptr_vector.size());
    for (size_t i = 0; i < v.size(); ++i)
      v.reinit(dh_ptr_vector[i].n_dofs());
  }

  template <typename Number>
  void
  resize_vector(dealii::LinearAlgebra::distributed::BlockVector<Number>& v) const
  {
    AssertDimension(v.n_blocks(), dh_ptr_vector.size());
    for (unsigned int i = 0; static_cast<size_t>(i) < v.n_blocks(); ++i)
    {
      mf.initialize_dof_vector(v.block(i), i);
      std::cout << "Vector " << i << " has size " << v.block(i).size() << std::endl;
    }
  }

  template <typename Number>
  void
  resize_vector(dealii::LinearAlgebra::distributed::Vector<Number>& v) const
  {
    AssertDimension(dh_ptr_vector.size(), 1);
    mf.initialize_dof_vector(v);
    std::cout << "Vector has size " << v.size() << std::endl;
  }

  template <typename Number, class Form>
  void
  vmult(dealii::LinearAlgebra::distributed::Vector<Number>& dst,
        const dealii::LinearAlgebra::distributed::Vector<Number>& src, Form& form) const
  {
    MatrixFreeIntegrator<dim, dealii::LinearAlgebra::distributed::Vector<Number>, Form, FEDatas>
      integrator;
    integrator.initialize(mf, std::make_shared<Form>(form), fe_datas);
    integrator.vmult(dst, src);
  }

  template <typename Number, class Form>
  void
  vmult(dealii::LinearAlgebra::distributed::BlockVector<Number>& dst,
        const dealii::LinearAlgebra::distributed::BlockVector<Number>& src, Form& form) const
  {
    MatrixFreeIntegrator<dim,
                         dealii::LinearAlgebra::distributed::BlockVector<Number>,
                         Form,
                         FEDatas>
      integrator;
    integrator.initialize(mf, std::make_shared<Form>(form), fe_datas);
    integrator.vmult(dst, src);
  }

  template <typename Number, class Form>
  void
  vmult_add(dealii::LinearAlgebra::distributed::BlockVector<Number>& dst,
            const dealii::LinearAlgebra::distributed::BlockVector<Number>& src, Form& form) const
  {
    MatrixFreeIntegrator<dim,
                         dealii::LinearAlgebra::distributed::BlockVector<Number>,
                         Form,
                         FEDatas>
      integrator(form, *fe_datas);
    integrator.initialize(mf);
    integrator.vmult_add(dst, src);
  }
};

template <int dim, class FEDatas>
MatrixFreeData<dim, FEDatas>
make_matrix_free_data(unsigned int grid_index, unsigned int refine,
                      std::vector<dealii::FiniteElement<dim>*> fe, FEDatas fe_datas)
{
  return MatrixFreeData<dim, FEDatas>(grid_index, refine, fe, fe_datas);
}

template <int dim, class FEData>
auto
make_matrix_free_data(unsigned int grid_index, unsigned int refine, dealii::FiniteElement<dim>& fe,
                      FEData& fe_data)
{
  std::vector<dealii::FiniteElement<dim>*> fes;
  fes.push_back(&fe);
  return MatrixFreeData<dim, FEDatas<FEData>>(
    grid_index, refine, fes, std::make_shared<FEDatas<FEData>>(fe_data));
}
#endif // MATRIXFREE_DATA_H
