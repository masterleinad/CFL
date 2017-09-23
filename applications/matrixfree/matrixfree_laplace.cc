// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include "matrixfree_data.h"
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <cfl/matrixfree/fefunctions.h>
#include <cfl/matrixfree/forms.h>

using namespace dealii;
using namespace CFL;
using namespace CFL::dealii::MatrixFree;

template <int dim>
void
test(unsigned int refine, unsigned int degree, const LinearAlgebra::distributed::Vector<double>& in,
     LinearAlgebra::distributed::Vector<double>& out)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(refine);

  FESystem<dim> fe(FE_Q<dim>(degree), dim);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  ConstraintMatrix constraints;
  constraints.close();

  MappingQ<dim, dim> mapping(degree);
  const unsigned int n_dofs = dof.n_dofs();

  SparsityPattern sparsity;
  {
    DynamicSparsityPattern csp(n_dofs, n_dofs);
    DoFTools::make_sparsity_pattern(dof, csp, constraints, true);
    sparsity.copy_from(csp);
  }
  SparseMatrix<double> sparse_matrix(sparsity);
  {
    QGauss<dim> quadrature_formula(degree + 1);

    FEValues<dim> fe_values(
      mapping, dof.get_fe(), quadrature_formula, update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = dof.get_fe().dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active(), endc = dof.end();
    for (; cell != endc; ++cell)
    {
      cell_matrix = 0;
      fe_values.reinit(cell);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          unsigned int component_i = fe.system_to_component_index(i).first;
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            unsigned int component_j = fe.system_to_component_index(j).first;
            if (component_i == component_j)
            {
              cell_matrix(i, j) += (fe_values.shape_grad(i, q_point) *
                                    fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));
            }
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, local_dof_indices, sparse_matrix);
    }
  }

  sparse_matrix.vmult(out, in);
  out.print(std::cout);
}

template <int dim>
void
run(unsigned int grid_index, unsigned int refine, unsigned int degree)
{
  auto fe_u = std::make_shared<FESystem<dim>>(FE_Q<dim>(degree), dim);

  FEData<FESystem, 1, dim, dim, 0, 1> fedata1(fe_u);
  FEDatas<decltype(fedata1)> fe_datas{ fedata1 };

  std::vector<FiniteElement<dim>*> fes;
  fes.push_back(&(*fe_u));

  constexpr Base::TestFunction<1, dim, 0> v;
  auto Dv = grad(v);
  constexpr Base::FEFunction<1, dim, 0> u;
  auto Du = grad(u);
  auto f = transform(Base::form(Du, Dv));

  MatrixFreeData<dim,
                 decltype(fe_datas),
                 decltype(f),
                 LinearAlgebra::distributed::BlockVector<double>>
    data(grid_index, refine, fes, fe_datas, f);

  LinearAlgebra::distributed::BlockVector<double> x_new(1), x_old(1), x_ref(1);
  LinearAlgebra::distributed::BlockVector<double> b(1);
  data.resize_vector(b);
  data.resize_vector(x_new);
  data.resize_vector(x_old);
  data.resize_vector(x_ref);

  for (size_t i = 0; i < b.n_blocks(); ++i)
  {
    for (types::global_dof_index j = 0; j < b.block(i).size(); ++j)
      b.block(i)[j] = j;
  }

  test<dim>(refine, degree, b.block(0), x_ref.block(0));

  for (unsigned int i = 0; i < 2; ++i)
  {
    x_old = x_new;
    for (size_t j = 0; j < b.n_blocks(); ++j)
      x_new.block(j) = 0.;
    for (size_t k = 0; k < b.n_blocks(); ++k)
    {
      for (types::global_dof_index j = 0; j < x_new.block(k).size(); ++j)
      {
        // if (x_new.block(k)[j] != 0.)
        std::cout << i << '\t' << j << '\t' << k << '\t' << b.block(k)[j] << std::endl;
      }
    }
    data.vmult(x_new, b);
    for (size_t k = 0; k < b.n_blocks(); ++k)
    {
      for (types::global_dof_index j = 0; j < x_new.block(k).size(); ++j)
      {
        // if (x_new.block(k)[j] != 0.)
        std::cout << i << '\t' << j << '\t' << k << '\t' << x_new.block(k)[j] << std::endl;
      }
    }
    if (i > 0)
    {
      for (size_t k = 0; k < b.n_blocks(); ++k)
      {
        x_old.block(k) -= x_new.block(k);
        x_ref.block(k) -= x_new.block(k);
        std::cout << i << " error: " << x_old.block(k).l2_norm() << std::endl;
        std::cout << i << " error_ref: " << x_ref.block(k).l2_norm() << std::endl;
        Assert(x_old.block(k).l2_norm() < 1.e-20 ||
                 x_old.block(k).l2_norm() < 1.e-6 * x_new.block(k).l2_norm(),
               ExcInternalError());
        Assert(x_ref.block(k).l2_norm() < 1.e-20 ||
                 x_ref.block(k).l2_norm() < 1.e-6 * x_new.block(k).l2_norm(),
               ExcInternalError());
      }
    }
    std::cout << std::endl;
  }
}

int
main(int /*argc*/, char** /*argv*/)
{
  deallog.depth_console(10);
  //::dealii::MultithreadInfo::set_thread_limit( (argc > 1) ? atoi(argv[1]) : 1);
  std::cout << ::dealii::MultithreadInfo::n_threads() << std::endl;
  try
  {
    const unsigned int refine = 0;
    const unsigned int degree = 1;
    run<2>(0, refine, degree);
  }
  catch (std::exception& exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
