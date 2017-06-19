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

#include <cfl/cfl.h>
#include <cfl/dealii_matrixfree.h>

// To generate a reference solution
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace CFL;
using namespace CFL::dealii::MatrixFree;

// Reference solution created with MeshWorker
template <int dim>
class MatrixIntegrator : public MeshWorker::LocalIntegrator<dim>
{
public:
  void cell(MeshWorker::DoFInfo<dim> &dinfo,
            typename MeshWorker::IntegrationInfo<dim> &info) const;
  void boundary(MeshWorker::DoFInfo<dim> &dinfo,
                typename MeshWorker::IntegrationInfo<dim> &info) const;
  void face(MeshWorker::DoFInfo<dim> &dinfo1,
            MeshWorker::DoFInfo<dim> &dinfo2,
            typename MeshWorker::IntegrationInfo<dim> &info1,
            typename MeshWorker::IntegrationInfo<dim> &info2) const;
};



template <int dim>
void MatrixIntegrator<dim>::cell(
  MeshWorker::DoFInfo<dim> &dinfo,
  typename MeshWorker::IntegrationInfo<dim> &info) const
{
  //LocalIntegrators::Laplace::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values());
}



template <int dim>
void MatrixIntegrator<dim>::face(
  MeshWorker::DoFInfo<dim> &dinfo1,
  MeshWorker::DoFInfo<dim> &dinfo2,
  typename MeshWorker::IntegrationInfo<dim> &info1,
  typename MeshWorker::IntegrationInfo<dim> &info2) const
{/*
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  // Manually compute penalty parameter instead of using the function
  // compute_penalty because we do it slightly differently on non-Cartesian
  // meshes.
  Tensor<2,dim> inverse_jacobian = transpose(info1.fe_values(0).jacobian(0).covariant_form());
  const double normal_volume_fraction1 = std::abs((inverse_jacobian[GeometryInfo<dim>::unit_normal_direction[dinfo1.face_number]]*info1.fe_values(0).normal_vector(0)));
  inverse_jacobian = transpose(info2.fe_values(0).jacobian(0).covariant_form());
  const double normal_volume_fraction2 = std::abs((inverse_jacobian[GeometryInfo<dim>::unit_normal_direction[dinfo2.face_number]]*info1.fe_values(0).normal_vector(0)));
  double penalty = 0.5*(normal_volume_fraction1+normal_volume_fraction2)*
                   std::max(1U,deg)*(deg+1.0);*/
  const double penalty = 1.;
  LocalIntegrators::Laplace
  ::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix,
              dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
              info1.fe_values(0), info2.fe_values(0), penalty);
}



template <int dim>
void MatrixIntegrator<dim>::boundary(
  MeshWorker::DoFInfo<dim> &dinfo,
  typename MeshWorker::IntegrationInfo<dim> &info) const
{/*
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Tensor<2,dim> inverse_jacobian = transpose(info.fe_values(0).jacobian(0).covariant_form());
  const double normal_volume_fraction = std::abs((inverse_jacobian[GeometryInfo<dim>::unit_normal_direction[dinfo.face_number]]*info.fe_values(0).normal_vector(0)));
  double penalty = normal_volume_fraction * std::max(1U,deg) * (deg + 1.0);
  LocalIntegrators::Laplace
  ::nitsche_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), penalty);*/
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
  info_box.initialize_gauss_quadrature(degree+1, degree+1, degree+1);
  info_box.initialize(dof.get_fe(), mapping);

  MeshWorker::DoFInfo<dim> dof_info(dof);

  MeshWorker::Assembler::MatrixSimple<SparseMatrix<double> > assembler;
  assembler.initialize(matrix);

  MatrixIntegrator<dim> integrator;
  MeshWorker::integration_loop<dim, dim>
  (dof.begin_active(), dof.end(), dof_info, info_box, integrator, assembler);

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
  auto fe_datas = (fedata1, fedata_face1);

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

  SumFEFunctions<FEFunctionExteriorFace<0, 1, 0>,
                 FEFunctionInteriorFace<0, 1, 0> > flux = u_p-u_m;
  auto flux_grad = Dnu_p-Dnu_m;
  auto jump = face_form(u_m-u_p, v_m)/*-face_form(flux, v_m)*/;

  auto flux1 = face_form(flux, Dnv_p)+face_form(flux, Dnv_m);
  auto flux2 = face_form(flux_grad, v_p)-face_form(flux_grad, v_m);

  auto boundary1 = boundary_form(u_p, v_p);
  auto boundary2 = boundary_form(Dnu_p, v_p);
  auto boundary3 = boundary_form(u_p, Dnv_p);

  auto face = jump/*+flux1+flux2*/;
  auto f = face/*+cell+boundary1+boundary2+boundary3*/;

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
      b.block(i)[j] = 1/*j*/;
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
        std::cout << i << '\t' << j << '\t' << k << '\t'
                  << '\t' << x_ref.block(k)[j]
                  << '\t' << x_new.block(k)[j] << std::endl;
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
