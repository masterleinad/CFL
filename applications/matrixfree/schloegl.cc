#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

using namespace dealii;

// TODO(darndt):
// nonlinear equations

// template <int dim>
// class BoundaryValues: public Function<dim>
// {
// public:
//   BoundaryValues()
//     : Function<dim>()
//   {}

//   double value (const Point<dim> &p, const unsigned int component) const
//   {
//     return std::sin(2*numbers::PI*(p(0)+p(1)));
//   }

// };

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
    return std::sin(numbers::PI * p(0)) * std::sin(numbers::PI * p(1));
  }

  Tensor<1, dim>
  gradient(const Point<dim>& p, [[maybe_unused]] const unsigned int component = 0) const override
  {
    Tensor<1, dim> ret_value;
    ret_value[0] = numbers::PI * std::cos(numbers::PI * p(0)) * std::sin(numbers::PI * p(1));
    ret_value[1] = numbers::PI * std::cos(numbers::PI * p(1)) * std::sin(numbers::PI * p(0));
    return ret_value;
  }

  double
  laplacian(const Point<dim>& p, [[maybe_unused]] const unsigned int component = 0) const override
  {
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
  value(const Point<dim>& p, const unsigned int /*component*/) const override
  {
    return -ref_func.laplacian(p) + ref_func.value(p) * ref_func.value(p) * ref_func.value(p) -
           alpha * ref_func.value(p);
  }

  ReferenceFunction<dim> ref_func;

private:
  double alpha;
};

template <int dim>
class Schloegl
{
public:
  Schloegl();

  ~Schloegl();

  void run(unsigned int cycles);
  void set_alpha(double a);

private:
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(unsigned int cycle);
  void create_mesh();

  FE_Q<dim> fe;
  MappingQ<dim> mapping;
  SphericalManifold<dim> manifold;
  Triangulation<dim> tria;
  DoFHandler<dim> dof_handler;

  SparsityPattern sp;
  AffineConstraints<double> constraints;
  SparseMatrix<double> system_matrix;
  Vector<double> system_rhs;
  Vector<double> solution;
  Vector<double> nwtn_update;
  double alpha{ 0. };
};

template <int dim>
Schloegl<dim>::Schloegl()
  : fe(1)
  , mapping(fe.degree)
  , dof_handler(tria)
{
  create_mesh();
}

template <int dim>
Schloegl<dim>::~Schloegl()
{
  tria.set_manifold(0);
}

template <int dim>
void
Schloegl<dim>::create_mesh()
{
  //    Point<dim> center;
  //    GridGenerator::hyper_shell  (tria, center, 0.5, 1., 0, false);
  //    tria.set_all_manifold_ids(0);
  //    static const SphericalManifold<dim> manifold(center);
  //    tria.set_manifold(0, manifold);

  // Point<dim> p1;
  // Point<dim> p2;
  // for (unsigned int d=0; d<dim; ++d)
  //   {
  //     p1(d)=-1.;
  //     p2(d)=1.;
  //   }
  // Assert(dim==2, ExcNotImplemented());

  // GridGenerator::hyper_ball<2>(tria);
  // tria.set_all_manifold_ids_on_boundary(0);

  // tria.set_manifold(0, manifold);

  GridGenerator::hyper_cube<dim>(tria, 0., 1.);
  //  tria.refine_global(1);

  GridOut grid_out;
  std::ofstream out("tria2.eps");
  grid_out.write_eps(tria, out);
}

template <int dim>
void
Schloegl<dim>::set_alpha(double a)
{
  alpha = a;
}

template <int dim>
void
Schloegl<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  // VectorTools::interpolate_boundary_values
  //   (dof_handler, 0, ZeroFunction<dim>(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sp.copy_from(dsp);
  // std::ofstream out("sparsity");
  // sp.print_svg(out);

  system_matrix.reinit(sp);
  system_rhs.reinit(dof_handler.n_dofs());
  solution.reinit(dof_handler.n_dofs());
  nwtn_update.reinit(dof_handler.n_dofs());

  //  std::srand(std::time(0));
  //  for ( unsigned int i = 0; i<dof_handler.n_dofs(); ++i )
  //    solution(i) =  (0.00001 * std::rand())/RAND_MAX + 0. /*- (3. * std::rand())/RAND_MAX*/ ;
  solution = 100.;

  // std::map<types::global_dof_index, double> boundary_values;
  // typename FunctionMap<dim>::type functions;
  // BoundaryValues<dim> b_values;
  // functions[0] = &b_values;
  // VectorTools::project_boundary_values(mapping, dof_handler, functions,
  //                                      QGauss<dim-1>(fe.degree+1), boundary_values);

  // std::map<types::global_dof_index, double>::iterator it = boundary_values.begin();
  // for (; it!=boundary_values.end(); ++it)
  //   {
  //     solution(it->first)=it->second;
  //   }

  // ReferenceFunction<dim> ref_function ;
  // VectorTools::project(mapping,dof_handler,constraints,QGauss<dim>(fe.degree+1),ref_function,solution);
}

template <int dim>
void
Schloegl<dim>::assemble_system()
{
  std::cout << "solution_before_assembly\n";
  solution.print(std::cout);

  QGauss<dim> quadrature(fe.get_degree() + 1);
  FEValues<dim> fev(mapping,
                    fe,
                    quadrature,
                    update_values | update_gradients | update_JxW_values |
                      update_quadrature_points);
  const unsigned int dpc = fe.dofs_per_cell;
  const unsigned int nqp = quadrature.size();
  FullMatrix<double> local_matrix(dpc);
  Vector<double> local_rhs(dpc);
  std::vector<types::global_dof_index> global_dof_idx(dpc);
  RHS<dim> rhs_function(alpha);

  std::vector<Tensor<1, dim>> old_local_gradients(nqp);
  std::vector<double> old_local_values(nqp);
  std::vector<double> rhs_values(nqp, 0.);

  system_matrix = 0.;
  system_rhs = 0.;

  ReferenceFunction<dim> ref_function;
  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell != endc; ++cell)
  {
    fev.reinit(cell);
    cell->get_dof_indices(global_dof_idx);
    local_matrix = 0.;
    local_rhs = 0.;

    fev.get_function_gradients(solution, old_local_gradients);
    fev.get_function_values(solution, old_local_values);
    // rhs_function.value_list(fev.get_quadrature_points(),rhs_values);

    for (unsigned int q = 0; q < nqp; ++q)
    {
      // const double coeff = 1./std::sqrt(1.+old_local_gradients[q].norm_square());
      for (unsigned int i = 0; i < dpc; ++i)
      {
        for (unsigned int j = 0; j < dpc; ++j)
        {
          // (linearized) laplacian
          local_matrix(i, j) += fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
          // (linearized) reaction
          local_matrix(i, j) += fev.shape_value(i, q) *
                                (3 * old_local_values[q] * old_local_values[q] - alpha) *
                                fev.shape_value(j, q) * fev.JxW(q);
        }
        // rhs wrt old_solution
        // double tmp1 = fev.shape_grad(i, q) * old_local_gradients[q];
        //        double tmp2a = fev.shape_value(i, q);
        //        double tmp2b = old_local_values[q];
        //        double tmp2c = alpha;
        //        double tmp3 = -fev.shape_value(i, q) * rhs_values[q];
        local_rhs(i) += -fev.JxW(q) * (fev.shape_grad(i, q) * old_local_gradients[q] +
                                       fev.shape_value(i, q) * old_local_values[q] *
                                         (old_local_values[q] * old_local_values[q] - alpha) -
                                       fev.shape_value(i, q) * rhs_values[q]);
        // std::cout << i << ": " << local_rhs(i) << std::endl;
        // std::cout << "tmp2a: " << tmp2a << std::endl;
        //        std::cout << "tmp2b: " << tmp2b << std::endl;
        //        std::cout << "tmp2c: " << tmp2c << std::endl;
        //        std::cout << "JxW: " << fev.JxW(q) << std::endl;

        // std::cout << "old_value[" << fev.quadrature_point(q) << "] " << old_local_values[q] <<
        // std::endl;
        // std::cout << "ref_value[" << fev.quadrature_point(q) << "] " <<
        // ref_function.value(fev.quadrature_point(q)) << std::endl;
        // std::cout << "old_grad[" << fev.quadrature_point(q) << "] " << old_local_gradients[q] <<
        // std::endl;
        // std::cout << "ref_grad[" << fev.quadrature_point(q) << "] " <<
        // ref_function.gradient(fev.quadrature_point(q)) << std::endl;
      }
    }
    local_rhs.print(std::cout);
    constraints.distribute_local_to_global(
      local_matrix, local_rhs, global_dof_idx, system_matrix, system_rhs);
  }
}

template <int dim>
void
Schloegl<dim>::solve()
{
  SolverControl solver_control(10000, 1.e-10, false, true);
  SolverCG<> solver(solver_control);
  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  std::cout << "rhs: \n";
  system_rhs.print(std::cout);
  solver.solve(system_matrix, nwtn_update, system_rhs, preconditioner);
  constraints.distribute(nwtn_update);
  const double b = .2;
  std::cout << "update: " << nwtn_update.l2_norm() << std::endl;
  nwtn_update *= b;
  solution += nwtn_update;
  std::cout << "solution: \n";
  solution.print(std::cout);
}

template <int dim>
void
Schloegl<dim>::output_results(unsigned int cycle)
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.add_data_vector(dof_handler, nwtn_update, "nwtn_update");
  data_out.build_patches(fe.get_degree());
  {
    std::ofstream out("solution." + Utilities::int_to_string(cycle) + ".vtu");
    data_out.write_vtu(out);
  }
}

template <int dim>
void
Schloegl<dim>::run(unsigned int cycles)
{
  setup_system();
  output_results(0);
  for (unsigned int cycle = 1; cycle < cycles; ++cycle)
  {
    std::cout << "cycle: " << cycle << std::endl;
    assemble_system();
    solve();
    output_results(cycle);
  }
}

int
main()
{
  Schloegl<2> Schloegl_2d;
  Schloegl_2d.set_alpha(10.);
  Schloegl_2d.run(2);
}
