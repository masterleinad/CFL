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
     LinearAlgebra::distributed::BlockVector<double>& out)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(refine);

  FE_Q<dim> fe_u_scal(degree + 1);
  FE_Q<dim> fe_p(degree);
  FESystem<dim> fe(fe_u_scal, dim, fe_p, 1);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  ConstraintMatrix constraints;
  constraints.close();

  MappingQ<dim, dim> mapping(degree);

  BlockSparsityPattern sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> solution;
  BlockVector<double> system_rhs;

  std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
  stokes_sub_blocks[dim] = 1;
  DoFRenumbering::component_wise(dof, stokes_sub_blocks);

  std::vector<types::global_dof_index> dofs_per_block(2);
  DoFTools::count_dofs_per_block(dof, dofs_per_block, stokes_sub_blocks);

  {
    BlockDynamicSparsityPattern csp(2, 2);

    for (unsigned int d = 0; d < 2; ++d)
      for (unsigned int e = 0; e < 2; ++e)
        csp.block(d, e).reinit(dofs_per_block[d], dofs_per_block[e]);

    csp.collect_sizes();

    DoFTools::make_sparsity_pattern(dof, csp, constraints, false);
    sparsity_pattern.copy_from(csp);
  }

  system_matrix.reinit(sparsity_pattern);

  // this is from step-22
  {
    QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim> fe_values(
      mapping, fe, quadrature_formula, update_values | update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<SymmetricTensor<2, dim>> phi_grads_u(dofs_per_cell);
    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active(), endc = dof.end();
    for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      local_matrix = 0;

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_grads_u[k] = fe_values[velocities].symmetric_gradient(k, q);
          div_phi_u[k] = fe_values[velocities].divergence(k, q);
          phi_p[k] = fe_values[pressure].value(k, q);
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(i, j) += (phi_grads_u[i] * phi_grads_u[j] + div_phi_u[i] * phi_p[j] +
                                   phi_p[i] * div_phi_u[j]) *
                                  fe_values.JxW(q);
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
    }
  }

  solution.reinit(2);
  for (unsigned int d = 0; d < 2; ++d)
    solution.block(d).reinit(dofs_per_block[d]);
  solution.collect_sizes();

  system_rhs.reinit(solution);

  // fill system_rhs with random numbers
  for (unsigned int i = 0; i < system_rhs.n_blocks(); ++i)
    for (unsigned int j = 0; j < system_rhs.block(i).size(); ++j)
      system_rhs.block(i)(j) = in.block(i)(j);

  system_rhs.print(std::cout);
  system_matrix.vmult(solution, system_rhs);

  for (unsigned int i = 0; i < system_rhs.n_blocks(); ++i)
    for (unsigned int j = 0; j < system_rhs.block(i).size(); ++j)
      out.block(i)(j) = solution.block(i)(j);

  solution.print(std::cout);
}

template <int dim>
void
run(unsigned int grid_index, unsigned int refine, unsigned int degree)
{
  FESystem<dim> fe_u(FE_Q<dim>(degree + 1), dim);
  FE_Q<dim> fe_p(degree);

  FEData<2, dim, dim, 0, 2> fedata1(fe_u);
  FEData<1, 1, dim, 1, 2> fedata2(fe_p);
  auto fe_datas = (fedata1, fedata2);

  std::vector<FiniteElement<dim>*> fes;
  fes.push_back(&fe_u);
  fes.push_back(&fe_p);
  // MatrixFreeData<dim, decltype(fe_datas) > data(grid_index, refine, fes, fe_datas);
  auto data = make_matrix_free_data(grid_index, refine, fes, fe_datas);
  data.initialize();

  TestFunction<1, dim, 0> v;
  TestFunction<0, dim, 1> q;
  FEFunction<1, dim, 0> u("u0");
  FEFunction<0, dim, 1> p("p");

  auto Dv = grad(v);
  //  TestSymmetricGradient<2,dim,0> Dv;
  auto Divu = div(u);
  FELiftDivergence<decltype(p)> Liftp(p);
  FESymmetricGradient<2, dim, 0> Du;
  auto f1 = form(Du + Liftp, Dv);
  auto f2 = form(Divu, q);
  auto f = f1 + f2;

  LinearAlgebra::distributed::BlockVector<double> x_new(2), x_old(2), x_ref(2);
  LinearAlgebra::distributed::BlockVector<double> b(2);
  data.resize_vector(b);
  data.resize_vector(x_new);
  data.resize_vector(x_old);
  data.resize_vector(x_ref);

  for (size_t i = 0; i < b.n_blocks(); ++i)
    for (types::global_dof_index j = 0; j < b.block(i).size(); ++j)
      b.block(i)[j] = j;

  test<dim>(refine, degree, b, x_ref);

  for (unsigned int i = 0; i < 2; ++i)
  {
    x_old = x_new;
    for (size_t j = 0; j < b.n_blocks(); ++j)
      x_new.block(j) = 0.;
    for (size_t k = 0; k < b.n_blocks(); ++k)
      for (types::global_dof_index j = 0; j < x_new.block(k).size(); ++j)
        // if (x_new.block(k)[j] != 0.)
        std::cout << i << '\t' << j << '\t' << k << '\t' << b.block(k)[j] << std::endl;
    data.vmult(x_new, b, f);
    for (size_t k = 0; k < b.n_blocks(); ++k)
      for (types::global_dof_index j = 0; j < x_new.block(k).size(); ++j)
        // if (x_new.block(k)[j] != 0.)
        std::cout << i << '\t' << j << '\t' << k << '\t' << x_new.block(k)[j] << '\t'
                  << x_ref.block(k)[j] << std::endl;
    if (i > 0)
    {
      for (size_t k = 0; k < b.n_blocks(); ++k)
      {
        x_old.block(k) -= x_new.block(k);
        x_ref.block(k) -= x_new.block(k);
        std::cout << i << " " << k << " error: " << x_old.block(k).l2_norm() << std::endl;
        std::cout << i << " " << k << " error_ref: " << x_ref.block(k).l2_norm() << std::endl;
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
