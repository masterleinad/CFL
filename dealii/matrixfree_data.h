#ifndef MATRIXFREE_DATA_H
#define MATRIXFREE_DATA_H

/*#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>



#include <deal.II/grid/tria.h>

#include <dealii/operators.h>

#include <deal.II/matrix_free/operators.h>


#include <cfl/forms.h>
#include <cfl/traits.h>
#include <cfl/static_for.h>

#include <string>*/

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>

#include <cfl/dealii_matrixfree.h>

#include <dealii/matrix_free_integrator.h>
#include <dealii/fe_data.h>

using namespace dealii;
using namespace CFL::dealii::MatrixFree;

template <int dim, class FEDatas>
class MatrixFreeData
{
  const MappingQ<dim, dim> mapping;
  SphericalManifold<dim> sphere;
  Triangulation<dim> tr;
  std::vector<std::unique_ptr<DoFHandler<dim>>> dh_ptr_vector;
  std::vector<const DoFHandler<dim>*> dh_const_ptr_vector;
  std::vector<std::unique_ptr<ConstraintMatrix>> constraint_ptr_vector;
  std::vector<const ConstraintMatrix*> constraint_const_ptr_vector;
  std::vector<Quadrature<1>> quadrature_vector;
  MatrixFree<dim, double> mf;
  std::shared_ptr<FEDatas> fe_datas;
  const Quadrature<1> quadrature;

public:
  // constructor for multiple FiniteElements
  MatrixFreeData(unsigned int grid_index, unsigned int refine,
                 const std::vector<FiniteElement<dim>*>& fe,
                 const std::shared_ptr<FEDatas>& fe_datas)
    : mapping(FEDatas::max_degree)
    , fe_datas(fe_datas)
    , quadrature(FEDatas::max_degree + 1)
  {
    AssertThrow(fe.size() > 0, ExcInternalError());
    if (grid_index == 0)
      GridGenerator::hyper_cube(tr);
    else if (grid_index == 1)
    {
      GridGenerator::hyper_ball(tr);
      tr.set_manifold(0, sphere);
      tr.set_all_manifold_ids(0);
    }
    else
      throw std::logic_error(std::string("Unknown grid index") + std::to_string(grid_index));
    tr.refine_global(refine);

    for (size_t i = 0; i < fe.size(); ++i)
    {
      dh_ptr_vector.push_back(std::unique_ptr<DoFHandler<dim>>(new DoFHandler<dim>()));
      dh_ptr_vector[i]->initialize(tr, *(fe[i]));
      dh_const_ptr_vector.push_back(dh_ptr_vector[i].get());
      // dh_vector[i].initialize_local_block_info();
      constraint_ptr_vector.push_back(std::unique_ptr<ConstraintMatrix>(new ConstraintMatrix()));
      constraint_const_ptr_vector.push_back(constraint_ptr_vector[i].get());
      quadrature_vector.push_back(QGauss<1>(fe[i]->degree + 1));
    }

    deallog << "Grid type " << grid_index << " Cells " << tr.n_active_cells() << " DoFs ";
    for (size_t i = 0; i < fe.size() - 1; ++i)
      deallog << dh_ptr_vector[i]->n_dofs() << "+";
    deallog << dh_ptr_vector[fe.size() - 1]->n_dofs() << std::endl;
  }

  // constructor for multiple FiniteElements
  MatrixFreeData(unsigned int grid_index, unsigned int refine,
                 const std::vector<FiniteElement<dim>*>& fe, FEDatas fe_datas)
    : MatrixFreeData(grid_index, refine, fe, std::make_shared<FEDatas>(fe_datas))
  {
  }

  void
  initialize()
  {
    mf.reinit(mapping, dh_const_ptr_vector, constraint_const_ptr_vector, quadrature_vector);
    fe_datas->initialize(mf);
  }

  void
  resize_vector(Vector<double>& v) const
  {
    AssertDimension(v.size(), dh_ptr_vector.size());
    for (size_t i = 0; i < v.size(); ++i)
      v.reinit(dh_ptr_vector[i].n_dofs());
  }

  template <typename Number>
  void
  resize_vector(LinearAlgebra::distributed::BlockVector<Number>& v) const
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
  resize_vector(LinearAlgebra::distributed::Vector<Number>& v) const
  {
    AssertDimension(dh_ptr_vector.size(), 1);
    mf.initialize_dof_vector(v);
    std::cout << "Vector has size " << v.size() << std::endl;
  }

  template <typename Number, class Form>
  void
  vmult(LinearAlgebra::distributed::Vector<Number>& dst,
        const LinearAlgebra::distributed::Vector<Number>& src, Form& form) const
  {
    MatrixFreeIntegrator<dim, Number, Form, FEDatas> integrator(form, *fe_datas);
    integrator.initialize(mf);
    integrator.vmult(dst, src);
  }

  template <typename Number, class Form>
  void
  vmult(LinearAlgebra::distributed::BlockVector<Number>& dst,
        const LinearAlgebra::distributed::BlockVector<Number>& src, Form& form) const
  {
    MatrixFreeIntegrator<dim, Number, Form, FEDatas> integrator(form, *fe_datas);
    integrator.initialize(mf);
    integrator.vmult(dst, src);
  }

  template <typename Number, class Form>
  void
  vmult_add(LinearAlgebra::distributed::BlockVector<Number>& dst,
            const LinearAlgebra::distributed::BlockVector<Number>& src, Form& form) const
  {
    MatrixFreeIntegrator<dim, Number, Form, FEDatas> integrator(form, *fe_datas);
    integrator.initialize(mf);
    integrator.vmult_add(dst, src);
  }
};

template <int dim, class FEDatas>
MatrixFreeData<dim, FEDatas>
make_matrix_free_data(unsigned int grid_index, unsigned int refine,
                      std::vector<FiniteElement<dim>*> fe, FEDatas fe_datas)
{
  return MatrixFreeData<dim, FEDatas>(grid_index, refine, fe, fe_datas);
}

template <int dim, class FEData>
auto make_matrix_free_data(unsigned int grid_index, unsigned int refine, FiniteElement<dim>& fe,
                           FEData& fe_data)
{
  std::vector<FiniteElement<dim>*> fes;
  fes.push_back(&fe);
  return MatrixFreeData<dim, FEDatas<FEData>>(
    grid_index, refine, fes, std::make_shared<FEDatas<FEData>>(fe_data));
}
#endif // MATRIXFREE_DATA_H
