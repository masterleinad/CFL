// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include "matrixfree_data.h"
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <cfl/cfl.h>
#include <cfl/dealii_matrixfree.h>

using namespace CFL;

template <int dim>
void
test(unsigned int refine, unsigned int degree,
     const LinearAlgebra::distributed::BlockVector<double>& in,
     LinearAlgebra::distributed::Vector<double>& out)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(refine);

  FE_Q<dim> fe(degree);
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
  SparseMatrix<double> system_matrix(sparsity);
  {
    const double alpha = 10.;
    QGauss<dim> quadrature_formula(degree + 1);

    FEValues<dim> fe_values(
      mapping, fe, quadrature_formula, update_values | update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> old_solution_local(n_q_points);

    typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active(), endc = dof.end();
    for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      fe_values.get_function_values(in.block(1), old_solution_local);
      local_matrix = 0;

      for (unsigned int q = 0; q < n_q_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            local_matrix(i, j) += (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) +
                                   fe_values.shape_value(i, q) * fe_values.shape_value(j, q) *
                                     (3 * std::pow(old_solution_local[q], 2.) - alpha)) *
                                  fe_values.JxW(q);

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
    }
  }

  system_matrix.vmult(out, in.block(0));
  out.print(std::cout);
}

template <int dim, unsigned int grid_index, unsigned int refine, unsigned int degree>
void
run()
{
  FE_Q<dim> fe(degree);

  FEData<degree, 1, dim, 0, degree, double> fedata_e(fe);
  FEData<degree, 1, dim, 1, degree, double> fedata_u(fe);
  auto fe_datas = (fedata_e, fedata_u);

  CFL::dealii::MatrixFree::TestFunction<0, dim, 0> v;
  auto Dv = grad(v);
  CFL::dealii::MatrixFree::FEFunction<0, dim, 0> e("e");
  CFL::dealii::MatrixFree::FEFunction<0, dim, 1> u("u");
  auto De = grad(e);

  const double alpha = 10;

  auto f1 = CFL::form(De, Dv);
  auto f2 = CFL::form(3 * u * u * e - alpha * e, v);
  auto f = f1 + f2;

  std::vector<FiniteElement<dim>*> fes;
  fes.push_back(&fe);
  fes.push_back(&fe);
  MatrixFreeData<dim, decltype(fe_datas)> data(grid_index, refine, fes, fe_datas);
  data.initialize();

  LinearAlgebra::distributed::BlockVector<double> in(2);
  LinearAlgebra::distributed::BlockVector<double> out(2);
  LinearAlgebra::distributed::Vector<double> ref;
  std::cout << in.n_blocks() << std::endl;
  data.resize_vector(in);
  data.resize_vector(out);
  ref = out.block(0);

  for (size_t i = 0; i < in.n_blocks(); ++i)
    for (types::global_dof_index j = 0; j < in.block(i).size(); ++j)
      in.block(i)[j] = j;

  in.print(std::cout);

  test<dim>(refine, degree, in, ref);
  {
    data.vmult(out, in, f);
    for (size_t k = 0; k < in.n_blocks(); ++k)
      for (types::global_dof_index j = 0; j < in.block(k).size(); ++j)
        std::cout << j << '\t' << k << '\t' << out.block(k)[j] << std::endl;

    ref -= out.block(0);
    std::cout << "error: " << ref.l2_norm() << std::endl;
    Assert(ref.l2_norm() < 1.e-20 || ref.l2_norm() < 1.e-6 * out.block(0).l2_norm(),
           ExcInternalError());
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
    constexpr int dim = 2;
    constexpr unsigned int grid_index = 0;
    constexpr unsigned int refine = 0;
    constexpr unsigned int degree = 1;
    run<dim, grid_index, refine, degree>();
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
