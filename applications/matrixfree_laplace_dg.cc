// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#define DEBUG_OUTPUT

#include "matrixfree_data.h"
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <matrixfree/fefunctions.h>

// To generate a reference solution
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include <fstream>

using namespace dealii;
using namespace CFL;
using namespace CFL::dealii::MatrixFree;

// Reference solution created with MeshWorker
template <int dim>
class MatrixIntegrator : public MeshWorker::LocalIntegrator<dim>
{
public:
  void cell(MeshWorker::DoFInfo<dim>& dinfo, typename MeshWorker::IntegrationInfo<dim>& info) const;
  void boundary(MeshWorker::DoFInfo<dim>& dinfo,
                typename MeshWorker::IntegrationInfo<dim>& info) const;
  void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
            typename MeshWorker::IntegrationInfo<dim>& info1,
            typename MeshWorker::IntegrationInfo<dim>& info2) const;
};

template <int dim>
void
MatrixIntegrator<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo,
                            typename MeshWorker::IntegrationInfo<dim>& info) const
{
  LocalIntegrators::Laplace::cell_matrix(dinfo.matrix(0, false).matrix, info.fe_values());
}

template <int dim>
void
MatrixIntegrator<dim>::face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                            typename MeshWorker::IntegrationInfo<dim>& info1,
                            typename MeshWorker::IntegrationInfo<dim>& info2) const
{
  /*
   const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
   // Manually compute penalty parameter instead of using the function
   // compute_penalty because we do it slightly differently on non-Cartesian
   // meshes.
   Tensor<2,dim> inverse_jacobian = transpose(info1.fe_values(0).jacobian(0).covariant_form());
   const double normal_volume_fraction1 =
   std::abs((inverse_jacobian[GeometryInfo<dim>::unit_normal_direction[dinfo1.face_number]]*info1.fe_values(0).normal_vector(0)));
   inverse_jacobian = transpose(info2.fe_values(0).jacobian(0).covariant_form());
   const double normal_volume_fraction2 =
   std::abs((inverse_jacobian[GeometryInfo<dim>::unit_normal_direction[dinfo2.face_number]]*info1.fe_values(0).normal_vector(0)));
   double penalty = 0.5*(normal_volume_fraction1+normal_volume_fraction2)*
                    std::max(1U,deg)*(deg+1.0);*/
  const double penalty = 1.;
  /*LocalIntegrators::Laplace
  ::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix,
              dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
              info1.fe_values(0), info2.fe_values(0), penalty);*/
  const double factor1 = 1.;
  const double factor2 = -1.;
  FullMatrix<double>& M11 = dinfo1.matrix(0, false).matrix;
  FullMatrix<double>& M12 = dinfo1.matrix(0, true).matrix;
  FullMatrix<double>& M21 = dinfo2.matrix(0, true).matrix;
  FullMatrix<double>& M22 = dinfo2.matrix(0, false).matrix;
  const FEValuesBase<dim>& fe1 = info1.fe_values(0);
  const FEValuesBase<dim>& fe2 = info2.fe_values(0);

  const unsigned int n_dofs = fe1.dofs_per_cell;
  /*AssertDimension(M11.n(), n_dofs);
  AssertDimension(M11.m(), n_dofs);
  AssertDimension(M12.n(), n_dofs);
  AssertDimension(M12.m(), n_dofs);
  AssertDimension(M21.n(), n_dofs);
  AssertDimension(M21.m(), n_dofs);
  AssertDimension(M22.n(), n_dofs);
  AssertDimension(M22.m(), n_dofs);*/

  const double nui = factor1;
  const double nue = (factor2 < 0) ? factor1 : factor2;
  const double nu = .5 * (nui + nue);

  for (unsigned int k = 0; k < fe1.n_quadrature_points; ++k)
  {
    const double dx = fe1.JxW(k);
    const Tensor<1, dim> n = fe1.normal_vector(k);
    for (unsigned int d = 0; d < fe1.get_fe().n_components(); ++d)
    {
      for (unsigned int i = 0; i < n_dofs; ++i)
      {
        for (unsigned int j = 0; j < n_dofs; ++j)
        {
          const double vi = fe1.shape_value_component(i, k, d);
          const double dnvi = n * fe1.shape_grad_component(i, k, d);
          const double ve = fe2.shape_value_component(i, k, d);
          const double dnve = n * fe2.shape_grad_component(i, k, d);
          const double ui = fe1.shape_value_component(j, k, d);
          const double dnui = n * fe1.shape_grad_component(j, k, d);
          const double ue = fe2.shape_value_component(j, k, d);
          const double dnue = n * fe2.shape_grad_component(j, k, d);
          M11(i, j) += dx * (-.5 * nui * dnvi * ui - .5 * nui * dnui * vi + nu * penalty * ui * vi);
          M12(i, j) += dx * (.5 * nui * dnvi * ue - .5 * nue * dnue * vi - nu * penalty * vi * ue);
          M21(i, j) += dx * (-.5 * nue * dnve * ui + .5 * nui * dnui * ve - nu * penalty * ui * ve);
          M22(i, j) += dx * (.5 * nue * dnve * ue + .5 * nue * dnue * ve + nu * penalty * ue * ve);
        }
      }
    }
  }
}

template <int dim>
void
MatrixIntegrator<dim>::boundary(MeshWorker::DoFInfo<dim>& dinfo,
                                typename MeshWorker::IntegrationInfo<dim>& info) const
{
  /*const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Tensor<2,dim> inverse_jacobian = transpose(info.fe_values(0).jacobian(0).covariant_form());
  const double normal_volume_fraction =
  std::abs((inverse_jacobian[GeometryInfo<dim>::unit_normal_direction[dinfo.face_number]]*info.fe_values(0).normal_vector(0)));
  double penalty = normal_volume_fraction * std::max(1U,deg) * (deg + 1.0);*/

  const double penalty = 1.;
  const double factor = 1.;
  const FEValuesBase<dim>& fe = info.fe_values(0);
  FullMatrix<double>& M = dinfo.matrix(0, false).matrix;

  const unsigned int n_dofs = fe.dofs_per_cell;
  const unsigned int n_comp = fe.get_fe().n_components();

  Assert(M.m() == n_dofs, ExcDimensionMismatch(M.m(), n_dofs));
  Assert(M.n() == n_dofs, ExcDimensionMismatch(M.n(), n_dofs));

  for (unsigned int k = 0; k < fe.n_quadrature_points; ++k)
  {
    const double dx = fe.JxW(k) * factor;
    const Tensor<1, dim> n = fe.normal_vector(k);
    for (unsigned int i = 0; i < n_dofs; ++i)
      for (unsigned int j = 0; j < n_dofs; ++j)
        for (unsigned int d = 0; d < n_comp; ++d)
          M(i, j) +=
            dx *
            (2. * fe.shape_value_component(i, k, d) * penalty * fe.shape_value_component(j, k, d) -
             (n * fe.shape_grad_component(i, k, d)) * fe.shape_value_component(j, k, d) -
             (n * fe.shape_grad_component(j, k, d)) * fe.shape_value_component(i, k, d));
  }
}

template <int dim>
void
test(unsigned int refine, unsigned int degree, const LinearAlgebra::distributed::Vector<double>& in,
     LinearAlgebra::distributed::Vector<double>& out)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(refine);

  FE_DGQ<dim> fe(degree);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);

  MappingQGeneric<dim, dim> mapping(degree);
  const unsigned int n_dofs = dof.n_dofs();

  // dof locations
  std::map<types::global_dof_index, Point<dim>> support_points;
  DoFTools::map_dofs_to_support_points(MappingQ<dim, dim>(1), dof, support_points);
  std::ofstream out_locations("dof_locations");
  DoFTools::write_gnuplot_dof_support_point_info(out_locations, support_points);

  SparsityPattern sparsity;
  {
    DynamicSparsityPattern csp(n_dofs, n_dofs);
    DoFTools::make_flux_sparsity_pattern(dof, csp);
    sparsity.copy_from(csp);
  }
  SparseMatrix<double> matrix(sparsity);

  MeshWorker::IntegrationInfoBox<dim> info_box;
  UpdateFlags update_flags = update_values | update_gradients | update_jacobians;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize_gauss_quadrature(degree + 1, degree + 1, degree + 1);
  info_box.initialize(dof.get_fe(), mapping);

  MeshWorker::DoFInfo<dim> dof_info(dof);

  MeshWorker::Assembler::MatrixSimple<SparseMatrix<double>> assembler;
  assembler.initialize(matrix);

  MatrixIntegrator<dim> integrator;
  MeshWorker::LoopControl loop_control;
  loop_control.own_faces = MeshWorker::LoopControl::both;
  MeshWorker::integration_loop<dim, dim>(
    dof.begin_active(), dof.end(), dof_info, info_box, integrator, assembler /*, loop_control*/);

  matrix.vmult(out, in);
  out.print(std::cout);
}

template <int dim, unsigned int degree>
void
run(unsigned int grid_index, unsigned int refine)
{
  FE_DGQ<dim> fe_u(degree);

  FEData<FE_DGQ, degree, 1, dim, 0, 1> fedata1(fe_u);
  FEDataFace<FE_DGQ, degree, 1, dim, 0, 1> fedata_face1(fe_u);
  auto fe_datas = (fedata_face1, fedata1);

  std::vector<FiniteElement<dim>*> fes;
  fes.push_back(&fe_u);

  TestFunction<0, 1, 0> v;
  auto Dv = grad(v);
  TestFunctionInteriorFace<0, 1, 0> v_p;
  TestFunctionExteriorFace<0, 1, 0> v_m;
  TestNormalGradientInteriorFace<0, 1, 0> Dnv_p;
  TestNormalGradientExteriorFace<0, 1, 0> Dnv_m;

  FEFunction<0, 1, 0> u("u");
  auto Du = grad(u);
  FEFunctionInteriorFace<0, 1, 0> u_p("u+");
  FEFunctionExteriorFace<0, 1, 0> u_m("u-");
  FENormalGradientInteriorFace<0, 1, 0> Dnu_p("u+");
  FENormalGradientExteriorFace<0, 1, 0> Dnu_m("u-");

  auto cell = form(Du, Dv);

  auto flux = u_p - u_m;
  auto flux_grad = Dnu_p - Dnu_m;

  auto flux1 = -face_form(flux, Dnv_p) + face_form(flux, Dnv_m);
  auto flux2 = face_form(-flux + .5 * flux_grad, v_p) - face_form(-flux + .5 * flux_grad, v_m);

  auto boundary1 = boundary_form(2. * u_p - Dnu_p, v_p);
  auto boundary3 = -boundary_form(u_p, Dnv_p);

  auto face = -flux2 + .5 * flux1;
  auto f = cell + face + boundary1 + boundary3;

  std::tuple<FormKind, unsigned int, IntegrationFlags> tuple =
    std::make_tuple(FormKind::cell, 0, IntegrationFlags());
  std::vector<std::tuple<FormKind, unsigned int, IntegrationFlags>> storage;
  storage.push_back(tuple);
  //  Assert((f.template check_forms<f.number, std::decay_t<IntegrationFlags>>()),
  //  ExcInternalError());

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

  for (unsigned int i = 0; i < 1; ++i)
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
        std::cout << i << '\t' << j << '\t' << k << '\t' << '\t' << x_ref.block(k)[j] << '\t'
                  << x_new.block(k)[j] << std::endl;
      }
      x_ref.block(k) -= x_new.block(k);
      std::cout << i << " error_ref: " << x_ref.block(k).l2_norm() << std::endl;
      Assert(x_ref.block(k).l2_norm() < 1.e-20 ||
               x_ref.block(k).l2_norm() < 1.e-6 * x_new.block(k).l2_norm(),
             ExcInternalError());
    }
    if (i > 0)
    {
      for (size_t k = 0; k < b.n_blocks(); ++k)
      {
        x_old.block(k) -= x_new.block(k);
        std::cout << i << " error: " << x_old.block(k).l2_norm() << std::endl;
        Assert(x_old.block(k).l2_norm() < 1.e-20 ||
                 x_old.block(k).l2_norm() < 1.e-6 * x_new.block(k).l2_norm(),
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
    constexpr unsigned int refine = 1;
    constexpr unsigned int degree = 1;
    run<2, degree>(0, refine);
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
