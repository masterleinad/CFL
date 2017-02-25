// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Katharina Kormann, Martin Kronbichler, Uppsala University,
 * 2009-2012, updated to MPI version with parallel vectors in 2016
 */

#include <dealii/mg_transfer_matrix_free.h>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include <cfl/dealii_matrixfree.h>
#include <cfl/forms.h>
#include <dealii/fe_data.h>
#include <dealii/matrix_free_integrator.h>

constexpr unsigned int degree_finite_element = 3;
constexpr unsigned int dimension = 2;
constexpr double alpha = 1.;

namespace Step37
{
using namespace dealii;

template <int dim>
class ReferenceFunction : public Function<dim>
{
public:
  ReferenceFunction()
    : Function<dim>(1)
  {
  }

  double
  value(const Point<dim>& p, [[maybe_unused]] const unsigned int component = 0) const override
  {
    Assert(component == 0, ExcInternalError());
    return std::sin(numbers::PI * p(0)) * std::sin(numbers::PI * p(1));
  }

  Tensor<1, dim>
  gradient(const Point<dim>& p, [[maybe_unused]] const unsigned int component = 0) const override
  {
    Assert(component == 0, ExcInternalError());
    Tensor<1, dim> ret_value;
    ret_value[0] = numbers::PI * std::cos(numbers::PI * p(0)) * std::sin(numbers::PI * p(1));
    ret_value[1] = numbers::PI * std::cos(numbers::PI * p(1)) * std::sin(numbers::PI * p(0));
    return ret_value;
  }

  double
  laplacian(const Point<dim>& p, [[maybe_unused]] const unsigned int component = 0) const override
  {
    Assert(component == 0, ExcInternalError());
    return -2 * numbers::PI * numbers::PI * std::sin(numbers::PI * p(0)) *
           std::sin(numbers::PI * p(1));
  }
};

template <int dim>
class RHS : public Function<dim>
{
public:
  explicit RHS(double a)
    : Function<dim>(1)
    , alpha(a)
  {
  }

  double
  value(const Point<dim>& p, const unsigned int /* component*/) const override
  {
    return -ref_func.laplacian(p) + ref_func.value(p) * ref_func.value(p) * ref_func.value(p) -
           alpha * ref_func.value(p);
  }

  ReferenceFunction<dim> ref_func;

private:
  double alpha;
};

template <int dim, class FEDatasSystem, class FEDatasLevel, class FormSystem, class FormRHS>
class LaplaceProblem
{
public:
  LaplaceProblem(const FEDatasSystem& mf_cfl_data_system_, const FEDatasLevel& mf_cfl_data_level_,
                 const FormSystem& form_system_, const FormRHS& form_rhs_);
  void run();

private:
  void setup_system();
  void assemble_rhs();
  double solve();
  void output_results(unsigned int cycle) const;

  const FEDatasSystem& mf_cfl_data_system;
  const FEDatasLevel& mf_cfl_data_level;
  const FormSystem& form_system;
  const FormRHS& form_rhs;

#ifdef DEAL_II_WITH_P4EST
  parallel::distributed::Triangulation<dim> triangulation;
#else
  Triangulation<dim> triangulation;
#endif

  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  std::vector<ConstraintMatrix> constraints;
  MatrixFree<dim, double> system_mf_storage;
  typedef MatrixFreeIntegrator<dim, double, FormSystem, FEDatasSystem> SystemMatrixType;
  SystemMatrixType system_matrix;
  typedef MatrixFreeIntegrator<dim, double, FormRHS, FEDatasSystem> RHSOperatorType;
  RHSOperatorType rhs_operator;

  MGLevelObject<MatrixFree<dim, float>> mg_mf_storage;
  typedef MatrixFreeIntegrator<dim, float, FormSystem, FEDatasLevel> LevelMatrixType;
  MGLevelObject<LevelMatrixType> mg_matrices;
  std::vector<MGConstrainedDoFs> mg_constrained_dofs;

  LinearAlgebra::distributed::BlockVector<double> solution;
  LinearAlgebra::distributed::BlockVector<double> solution_update;
  LinearAlgebra::distributed::BlockVector<double> system_rhs;

  double setup_time{};
  ConditionalOStream pcout;
  ConditionalOStream time_details;
};

template <int dim, class FEDatasSystem, class FEDatasLevel, class FormSystem, class FormRHS>
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, FormSystem, FormRHS>::LaplaceProblem(
  const FEDatasSystem& mf_cfl_data_system_, const FEDatasLevel& mf_cfl_data_level_,
  const FormSystem& form_system_, const FormRHS& form_rhs_)
  : mf_cfl_data_system(mf_cfl_data_system_)
  , mf_cfl_data_level(mf_cfl_data_level_)
  , form_system(form_system_)
  , form_rhs(form_rhs_)
  ,
#ifdef DEAL_II_WITH_P4EST
  triangulation(MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
  ,
#else
  triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
  ,
#endif
  fe(degree_finite_element)
  , dof_handler(triangulation)
  , constraints(2)
  , system_matrix()
  , mg_constrained_dofs(2)
  , solution(2)
  , solution_update(2)
  , system_rhs(2)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , time_details(std::cout, false)
{
}

template <int dim, class FEDatasSystem, class FEDatasLevel, class FormSystem, class FormRHS>
void
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, FormSystem, FormRHS>::setup_system()
{
  Timer time;
  time.start();
  setup_time = 0;

  system_matrix.clear();
  rhs_operator.clear();
  mg_matrices.clear_elements();

  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs(fe);

  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  constraints[0].clear();
  constraints[0].reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints[0]);
  VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(), constraints[0]);
  VectorTools::interpolate_boundary_values(dof_handler, 1, ZeroFunction<dim>(), constraints[0]);
  VectorTools::interpolate_boundary_values(dof_handler, 2, ZeroFunction<dim>(), constraints[0]);
  VectorTools::interpolate_boundary_values(dof_handler, 3, ZeroFunction<dim>(), constraints[0]);
  constraints[0].close();
  constraints[1].clear();
  constraints[1].reinit(locally_relevant_dofs);
  constraints[1].close();

  pcout << "n_constraints: " << constraints[0].n_constraints() << std::endl;

  setup_time += time.wall_time();
  time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time() << "s/" << time.wall_time()
               << "s" << std::endl;
  time.restart();
  {
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);

    std::vector<const DoFHandler<dim>*> dh_pointers;
    dh_pointers.push_back(&dof_handler);
    dh_pointers.push_back(&dof_handler);
    std::vector<const ConstraintMatrix*> constraints_pointers;
    constraints_pointers.push_back(&(constraints[0]));
    constraints_pointers.push_back(&(constraints[1]));

    std::vector<QGauss<1>> quadrature_pointers(2, QGauss<1>(fe.degree + 1));

    system_mf_storage.reinit(
      dh_pointers, constraints_pointers, quadrature_pointers, additional_data);
  }

  system_matrix.initialize(system_mf_storage,
                           std::make_shared<FormSystem>(form_system),
                           std::make_shared<FEDatasSystem>(mf_cfl_data_system));

  rhs_operator.initialize(system_mf_storage,
                          std::make_shared<FormRHS>(form_rhs),
                          std::make_shared<FEDatasSystem>(mf_cfl_data_system));

  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(system_rhs);
  system_matrix.initialize_dof_vector(solution_update);

  std::srand(std::time(nullptr));
  for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    solution(i) = ((2. * std::rand()) / RAND_MAX - 1.) * alpha;
  // solution = alpha;
  constraints[0].distribute(solution);

  setup_time += time.wall_time();
  time_details << "Setup matrix-free system   (CPU/wall) " << time() << "s/" << time.wall_time()
               << "s" << std::endl;
  time.restart();

  const unsigned int nlevels = triangulation.n_global_levels();
  mg_matrices.resize(0, nlevels - 1);
  mg_mf_storage.resize(0, nlevels - 1);

  std::set<types::boundary_id> dirichlet_boundary{ 0, 1, 2, 3 };
  mg_constrained_dofs[0].initialize(dof_handler);
  mg_constrained_dofs[0].make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
  mg_constrained_dofs[1].initialize(dof_handler);

  for (unsigned int level = 0; level < nlevels; ++level)
  {
    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
    std::vector<ConstraintMatrix> level_constraints(2);
    level_constraints[0].reinit(relevant_dofs);
    level_constraints[0].add_lines(mg_constrained_dofs[0].get_boundary_indices(level));
    level_constraints[0].close();
    level_constraints[1].reinit(relevant_dofs);
    //    level_constraints[1].add_lines(mg_constrained_dofs[1].get_boundary_indices(level));
    level_constraints[1].close();

    typename MatrixFree<dim, float>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);
    additional_data.level_mg_handler = level;

    std::vector<const DoFHandler<dim>*> dh_pointers;
    dh_pointers.push_back(&dof_handler);
    dh_pointers.push_back(&dof_handler);
    std::vector<const ConstraintMatrix*> constraints_pointers;
    constraints_pointers.push_back(&(level_constraints[0]));
    constraints_pointers.push_back(&(level_constraints[1]));

    std::vector<QGauss<1>> quadrature_pointers(2, QGauss<1>(fe.degree + 1));

    mg_mf_storage[level].reinit(
      dh_pointers, constraints_pointers, quadrature_pointers, additional_data);

    mg_matrices[level].initialize(mg_mf_storage[level],
                                  mg_constrained_dofs,
                                  level,
                                  std::make_shared<FormSystem>(form_system),
                                  std::make_shared<FEDatasLevel>(mf_cfl_data_level));
  }
  setup_time += time.wall_time();
  time_details << "Setup matrix-free levels   (CPU/wall) " << time() << "s/" << time.wall_time()
               << "s" << std::endl;
}

template <int dim, class FEDatasSystem, class FEDatasLevel, class FormSystem, class FormRHS>
void
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, FormSystem, FormRHS>::assemble_rhs()
{
  Timer time;

  QGauss<dim> quadrature(fe.get_degree() + 1);
  FEValues<dim> fev(fe,
                    quadrature,
                    update_values | update_gradients | update_JxW_values |
                      update_quadrature_points);
  const unsigned int dpc = fe.dofs_per_cell;
  const unsigned int nqp = quadrature.size();
  Vector<double> local_rhs(dpc);
  std::vector<types::global_dof_index> global_dof_idx(dpc);
  RHS<dim> rhs_function(alpha);

  std::vector<Tensor<1, dim>> old_local_gradients(nqp);
  std::vector<double> old_local_values(nqp);
  std::vector<double> rhs_values(nqp, 0.);

  system_rhs = 0.;

  LinearAlgebra::distributed::BlockVector<double> system_rhs_new = system_rhs;

  ReferenceFunction<dim> ref_function;
  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell != endc; ++cell)
  {
    fev.reinit(cell);
    cell->get_dof_indices(global_dof_idx);
    local_rhs = 0.;

    fev.get_function_gradients(solution.block(0), old_local_gradients);
    fev.get_function_values(solution.block(0), old_local_values);
    //      rhs_function.value_list(fev.get_quadrature_points(),rhs_values);

    for (unsigned int q = 0; q < nqp; ++q)
    {
      for (unsigned int i = 0; i < dpc; ++i)
      {
        local_rhs(i) += -fev.JxW(q) * (fev.shape_grad(i, q) * old_local_gradients[q] +
                                       fev.shape_value(i, q) * old_local_values[q] *
                                         (old_local_values[q] * old_local_values[q] - alpha) -
                                       fev.shape_value(i, q) * rhs_values[q]);
      }
    }
    constraints[0].distribute_local_to_global(local_rhs, global_dof_idx, system_rhs_new.block(0));
  }
  system_rhs_new.compress(VectorOperation::add);

  Assert(system_rhs_new.block(1).l2_norm() < 1e-10, ExcInternalError());

  solution.block(1) = solution.block(0);
  rhs_operator.vmult(system_rhs, solution);

  system_rhs_new -= system_rhs;
  Assert(system_rhs_new.l2_norm() < 1.e-10 * system_rhs.l2_norm(), ExcInternalError());

  setup_time += time.wall_time();
  time_details << "Assemble right hand side   (CPU/wall) " << time() << "s/" << time.wall_time()
               << "s" << std::endl;
}

template <int dim, class FEDatasSystem, class FEDatasLevel, class FormSystem, class FormRHS>
double
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, FormSystem, FormRHS>::solve()
{
  Timer time;
  MGTransferBlockMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
  mg_transfer.build_matrices(dof_handler);
  /*    setup_time += time.wall_time();
      time_details << "MG build transfer time     (CPU/wall) " << time() << "s/" << time.wall_time()
                   << "s\n";
      time.restart();

      typedef PreconditionChebyshev<LevelMatrixType, LinearAlgebra::distributed::BlockVector<float>>
        SmootherType;
      mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::BlockVector<float>>
      mg_smoother;
      MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
      smoother_data.resize(0, triangulation.n_global_levels() - 1);
      for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
      {
        if (level > 0)
        {
          smoother_data[level].smoothing_range = 15.;
          smoother_data[level].degree = 4;
          smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
          smoother_data[0].smoothing_range = 1e-3;
          smoother_data[0].degree = numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
        mg_matrices[level].compute_diagonal();
        //smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
      }
      mg_smoother.initialize(mg_matrices, smoother_data);

      MGCoarseGridApplySmoother<LinearAlgebra::distributed::BlockVector<float>> mg_coarse;
      mg_coarse.initialize(mg_smoother);

      mg::Matrix<LinearAlgebra::distributed::BlockVector<float>> mg_matrix(mg_matrices);

      MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
     mg_interface_matrices;
      mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
      for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_interface_matrices[level].initialize(mg_matrices[level]);
      mg::Matrix<LinearAlgebra::distributed::BlockVector<float>>
     mg_interface(mg_interface_matrices);

      Multigrid<LinearAlgebra::distributed::BlockVector<float>> mg(
        mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
      mg.set_edge_matrices(mg_interface, mg_interface);

      PreconditionMG<dim, LinearAlgebra::distributed::BlockVector<float>,
      MGTransferPrebuilt<LinearAlgebra::distributed::BlockVector<float> > >
        preconditioner(dof_handler, mg, mg_transfer);*/

  SolverControl solver_control(dof_handler.n_dofs(), 1e-12 * system_rhs.l2_norm(), false, false);
  SolverCG<LinearAlgebra::distributed::BlockVector<double>> cg(solver_control);
  setup_time += time.wall_time();
  time_details << "MG build smoother time     (CPU/wall) " << time() << "s/" << time.wall_time()
               << "s\n";
  pcout << "Total setup time               (wall) " << setup_time << "s\n";

  time.reset();
  time.start();

  // set nonlinearity to use
  solution.block(1) = solution.block(0);
  std::vector<bool> nonlinear_components;
  nonlinear_components.push_back(false);
  nonlinear_components.push_back(true);
  system_matrix.set_nonlinearities(nonlinear_components, solution);

  cg.solve(system_matrix, solution_update, system_rhs, /*preconditioner*/ PreconditionIdentity());

  constraints[0].distribute(solution_update.block(0));
  const double b = .2;
  solution_update *= b;
  solution += solution_update;
  pcout << "update: " << solution_update.l2_norm() << std::endl;
  pcout << "solution: " << solution.l2_norm() << std::endl;

  pcout << "Time solve (" << solver_control.last_step() << " iterations)  (CPU/wall) " << time()
        << "s/" << time.wall_time() << "s\n";
  return solution_update.l2_norm();
}

template <int dim, class FEDatasSystem, class FEDatasLevel, class FormSystem, class FormRHS>
void
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, FormSystem, FormRHS>::output_results(
  const unsigned int cycle) const
{
  if (triangulation.n_global_active_cells() > 1000000)
    return;

  DataOut<dim> data_out;

  solution.update_ghost_values();
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution.block(0), "solution");
  data_out.build_patches(degree_finite_element);

  std::ostringstream filename;
  filename << "solution-" << cycle << "." << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
           << ".vtu";

  std::ofstream output(filename.str().c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
    {
      std::ostringstream filename;
      filename << "solution-" << cycle << "." << i << ".vtu";

      filenames.emplace_back(filename.str());
    }
    std::string master_name = "solution-" + Utilities::to_string(cycle) + ".pvtu";
    std::ofstream master_output(master_name.c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

template <int dim, class FEDatasSystem, class FEDatasLevel, class FormSystem, class FormRHS>
void
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, FormSystem, FormRHS>::run()
{
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(6);
  setup_system();
  output_results(0);
  unsigned int cycle = 0;
  while (true)
  {
    ++cycle;
    pcout << "Cycle " << cycle << std::endl;
    assemble_rhs();
    const double update_size = solve();
    output_results(cycle);
    if (update_size < 1.e-3 * solution.l2_norm() || solution.l2_norm() < 1.e-10)
      break;
    pcout << std::endl;
  }
}
} // namespace Step37

int
main(int argc, char* argv[])
{
  deallog.depth_console(10);

  try
  {
    using namespace Step37;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

    FE_Q<dimension> fe_u(degree_finite_element);

    FEData<degree_finite_element, 1, dimension, 0, degree_finite_element, double> fedata_e_system(
      fe_u);
    FEData<degree_finite_element, 1, dimension, 1, degree_finite_element, double> fedata_u_system(
      fe_u);
    auto fe_datas_system = (fedata_e_system, fedata_u_system);
    FEData<degree_finite_element, 1, dimension, 0, degree_finite_element, float> fedata_e_level(
      fe_u);
    FEData<degree_finite_element, 1, dimension, 1, degree_finite_element, float> fedata_u_level(
      fe_u);
    auto fe_datas_level = (fedata_e_level, fedata_u_level);

    CFL::dealii::MatrixFree::TestFunction<0, dimension, 0> v;
    auto Dv = grad(v);
    CFL::dealii::MatrixFree::FEFunction<0, dimension, 0> e("e");
    auto De = grad(e);
    CFL::dealii::MatrixFree::FEFunction<0, dimension, 1> u("u");
    auto Du = grad(u);

    auto f1 = CFL::form(De, Dv);
    auto f2 = CFL::form(3 * u * u * e - alpha * e, v);
    auto f = f1 + f2;

    auto rhs = CFL::form(-Du, Dv) + CFL::form(-u * u * u + alpha * u, v);

    LaplaceProblem<dimension,
                   decltype(fe_datas_system),
                   decltype(fe_datas_level),
                   decltype(f),
                   decltype(rhs)>
      laplace_problem(fe_datas_system, fe_datas_level, f, rhs);
    laplace_problem.run();
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
