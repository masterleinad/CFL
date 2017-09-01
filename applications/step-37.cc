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

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/fe/fe_q.h>
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

#include <matrixfree/fefunctions.h>
#include <matrixfree/forms.h>
#include <matrixfree/fe_data.h>
#include <matrixfree/matrix_free_integrator.h>

const unsigned int degree_finite_element = 2;
const unsigned int dimension = 3;

namespace Step37
{
using namespace dealii;

template <int dim, class FEDatasSystem, class FEDatasLevel, class Form>
class LaplaceProblem
{
public:
  LaplaceProblem(FEDatasSystem& mf_cfl_data_system_, FEDatasLevel& mf_cfl_data_level_, Form& form_);
  void run();

private:
  void setup_system();
  void assemble_rhs();
  void solve();
  void output_results(const unsigned int cycle) const;

  FEDatasSystem& mf_cfl_data_system;
  FEDatasLevel& mf_cfl_data_level;
  Form& form;

#ifdef DEAL_II_WITH_P4EST
  parallel::distributed::Triangulation<dim> triangulation;
#else
  Triangulation<dim> triangulation;
#endif

  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  ConstraintMatrix constraints;
  MatrixFree<dim, double> system_mf_storage;
  typedef MatrixFreeIntegrator<dim, LinearAlgebra::distributed::Vector<double>, Form, FEDatasSystem>
    SystemMatrixType;
  SystemMatrixType system_matrix;

  MGLevelObject<MatrixFree<dim, float>> mg_mf_storage;
  typedef MatrixFreeIntegrator<dim, LinearAlgebra::distributed::Vector<float>, Form, FEDatasLevel>
    LevelMatrixType;
  MGLevelObject<LevelMatrixType> mg_matrices;
  MGConstrainedDoFs mg_constrained_dofs;

  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> solution_update;
  LinearAlgebra::distributed::Vector<double> system_rhs;

  double setup_time{};
  ConditionalOStream pcout;
  ConditionalOStream time_details;
};

template <int dim, class FEDatasSystem, class FEDatasLevel, class Form>
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, Form>::LaplaceProblem(
  FEDatasSystem& mf_cfl_data_system_, FEDatasLevel& mf_cfl_data_level_, Form& form_)
  : mf_cfl_data_system(mf_cfl_data_system_)
  , mf_cfl_data_level(mf_cfl_data_level_)
  , form(form_)
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
  , system_matrix()
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , time_details(std::cout, false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{
}

template <int dim, class FEDatasSystem, class FEDatasLevel, class Form>
void
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, Form>::setup_system()
{
  Timer time;
  time.start();
  setup_time = 0;

  system_matrix.clear();
  mg_matrices.clear_elements();

  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs(fe);

  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(), constraints);
  constraints.close();
  setup_time += time.wall_time();
  time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time() << "s/"
               << time.wall_time() << "s" << std::endl;
  time.restart();
  {
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);
    system_mf_storage.reinit(dof_handler, constraints, QGauss<1>(fe.degree + 1), additional_data);
  }

  system_matrix.initialize(std::make_shared<MatrixFree<dim, double>>(system_mf_storage),
                           std::make_shared<Form>(form),
                           std::make_shared<FEDatasSystem>(mf_cfl_data_system));

  //    system_matrix.evaluate_coefficient(Coefficient<dim>());

  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(system_rhs);

  setup_time += time.wall_time();
  time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time() << "s/"
               << time.wall_time() << "s" << std::endl;
  time.restart();

  const unsigned int nlevels = triangulation.n_global_levels();
  mg_matrices.resize(0, nlevels - 1);
  mg_mf_storage.resize(0, nlevels - 1);

  std::set<types::boundary_id> dirichlet_boundary;
  dirichlet_boundary.insert(0);
  mg_constrained_dofs.initialize(dof_handler);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

  for (unsigned int level = 0; level < nlevels; ++level)
  {
    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
    ConstraintMatrix level_constraints;
    level_constraints.reinit(relevant_dofs);
    level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
    level_constraints.close();

    typename MatrixFree<dim, float>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);
    additional_data.level_mg_handler = level;

    mg_mf_storage[level].reinit(
      dof_handler, level_constraints, QGauss<1>(fe.degree + 1), additional_data);

    mg_matrices[level].initialize(std::make_shared<MatrixFree<dim, float>>(mg_mf_storage[level]),
                                  mg_constrained_dofs,
                                  level,
                                  std::make_shared<Form>(form),
                                  std::make_shared<FEDatasLevel>(mf_cfl_data_level));
    // mg_matrices[level].evaluate_coefficient(Coefficient<dim>());
  }
  setup_time += time.wall_time();
  time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time() << "s/"
               << time.wall_time() << "s" << std::endl;
}

template <int dim, class FEDatasSystem, class FEDatasLevel, class Form>
void
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, Form>::assemble_rhs()
{
  Timer time;

  system_rhs = 0;
  FEEvaluation<dim, degree_finite_element> phi(system_mf_storage);
  for (unsigned int cell = 0; cell < system_mf_storage.n_macro_cells(); ++cell)
  {
    phi.reinit(cell);
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      phi.submit_value(make_vectorized_array<double>(1.0), q);
    phi.integrate(true, false);
    phi.distribute_local_to_global(system_rhs);
  }
  system_rhs.compress(VectorOperation::add);

  setup_time += time.wall_time();
  time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time() << "s/"
               << time.wall_time() << "s" << std::endl;
}

template <int dim, class FEDatasSystem, class FEDatasLevel, class Form>
void
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, Form>::solve()
{
  Timer time;
  MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler);
  setup_time += time.wall_time();
  time_details << "MG build transfer time     (CPU/wall) " << time.cpu_time() << "s/"
               << time.wall_time() << "s\n";
  time.restart();

  typedef PreconditionChebyshev<LevelMatrixType, LinearAlgebra::distributed::Vector<float>>
    SmootherType;
  mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
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
    smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
  }
  mg_smoother.initialize(mg_matrices, smoother_data);

  MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
  mg_coarse.initialize(mg_smoother);

  mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices);

  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>> mg_interface_matrices;
  mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
  for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    mg_interface_matrices[level].initialize(mg_matrices[level]);
  mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

  Multigrid<LinearAlgebra::distributed::Vector<float>> mg(
    mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  mg.set_edge_matrices(mg_interface, mg_interface);

  PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>, MGTransferMatrixFree<dim, float>>
    preconditioner(dof_handler, mg, mg_transfer);

  SolverControl solver_control(100, 1e-12 * system_rhs.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
  setup_time += time.wall_time();
  time_details << "MG build smoother time     (CPU/wall) " << time.cpu_time() << "s/"
               << time.wall_time() << "s\n";
  pcout << "Total setup time               (wall) " << setup_time << "s\n";

  time.reset();
  time.start();
  cg.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);

  pcout << "Time solve (" << solver_control.last_step() << " iterations)  (CPU/wall) "
        << time.cpu_time() << "s/" << time.wall_time() << "s\n";
}

template <int dim, class FEDatasSystem, class FEDatasLevel, class Form>
void
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, Form>::output_results(
  const unsigned int cycle) const
{
  if (triangulation.n_global_active_cells() > 1000000)
    return;

  DataOut<dim> data_out;

  solution.update_ghost_values();
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  {
    std::ostringstream filename;
    filename << "solution-" << cycle << "." << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
             << ".vtu";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtu(output);
  }

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

template <int dim, class FEDatasSystem, class FEDatasLevel, class Form>
void
LaplaceProblem<dim, FEDatasSystem, FEDatasLevel, Form>::run()
{
  for (unsigned int cycle = 0; cycle < 8 - dim; ++cycle)
  {
    pcout << "Cycle " << cycle << std::endl;

    if (cycle == 0)
    {
      GridGenerator::hyper_cube(triangulation, 0., 1.);
      triangulation.refine_global(3 - dim);
    }
    triangulation.refine_global(1);
    setup_system();
    assemble_rhs();
    solve();
    // output_results(cycle);
    pcout << std::endl;
  };
}
} // namespace Step37

int
main(int argc, char* argv[])
{
  try
  {
    using namespace Step37;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

    FE_Q<dimension> fe_u(degree_finite_element);

    FEData<FE_Q, 2, 1, dimension, 0, 2, double> fedata_double(fe_u);
    FEDatas<decltype(fedata_double)> fe_datas_system{ fedata_double };
    FEData<FE_Q, 2, 1, dimension, 0, 2, float> fedata_float(fe_u);
    FEDatas<decltype(fedata_float)> fe_datas_level{ fedata_float };

    CFL::dealii::MatrixFree::TestFunction<0, dimension, 0> v_system;
    auto Dv_system = grad(v_system);
    CFL::dealii::MatrixFree::FEFunction<0, dimension, 0> u_system("u");
    auto Du_system = grad(u_system);
    auto f_system = CFL::form(Du_system, Dv_system);

    LaplaceProblem<dimension,
                   decltype(fe_datas_system),
                   decltype(fe_datas_level),
                   decltype(f_system)>
      laplace_problem(fe_datas_system, fe_datas_level, f_system);
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
