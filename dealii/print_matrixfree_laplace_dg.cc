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
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include <fstream>

using namespace dealii;
using namespace CFL;
using namespace CFL::dealii::MatrixFree;


template<typename... Types>
class LatexForms;

template<typename Type, typename... Types>
class LatexForms<Type, Types...> : public LatexForms<Types...>
{
  public:

  std::string print()
  {
  int     status;
    return std::string(abi::__cxa_demangle(typeid(Type).name(), 0, 0, &status))+"\n"+LatexForms<Types...>::print();
  }
};

template<typename Type>
class LatexForms<Type>
{
  public:

  std::string print()
  {
int     status;
    return std::string(abi::__cxa_demangle(typeid(Type).name(), 0, 0, &status));
  }
};

template<typename... Types>
LatexForms<Types...> transform_latex(const Forms<Types...>&)
{
  return LatexForms<Types...>();
}

template <int dim, unsigned int degree>
void
run()
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

  std::cout << transform_latex(f).print() << std::endl;
}

int
main(int /*argc*/, char** /*argv*/)
{
  deallog.depth_console(10);
  //::dealii::MultithreadInfo::set_thread_limit( (argc > 1) ? atoi(argv[1]) : 1);
  //std::cout << ::dealii::MultithreadInfo::n_threads() << std::endl;
  try
  {
    constexpr unsigned int degree = 1;
    run<2, degree>();
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
