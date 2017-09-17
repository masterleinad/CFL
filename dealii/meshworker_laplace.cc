// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include "meshworker_data.h"
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>

#include <cfl/cfl.h>
#include <cfl/dealii_meshworker.h>

using namespace CFL;
using namespace CFL::dealii::MeshWorker;

template <int dim>
void
run(unsigned int grid_index, unsigned int refine, unsigned int degree)
{
  FE_Q<dim> fe(degree);
  MeshworkerData<dim> data(grid_index, refine, fe);

  TestFunction<0, dim> v(0, 0);
  FEFunction<0, dim> u(0, 0);
  auto Dv = grad(v);
  auto Du = grad(u);
  auto f = form(Du, Dv);

  Vector<double> x_new, x_old;
  Vector<double> b;
  data.resize_vector(x_old);
  data.resize_vector(x_new);
  data.resize_vector(b);

  for (unsigned int i = 0; i < b.size(); ++i)
    b[i] = i;

  for (unsigned int i = 0; i < 2; ++i)
  {
    x_old = x_new;
    x_new = 0.;
    data.vmult(x_new, b, f);
    for (unsigned int j = 0; j < x_new.size(); ++j)
    {
      if (x_new[j] != 0.)
        std::cout << i << '\t' << j << '\t' << x_new[j] << std::endl;
    }
    Assert(i == 0 || x_new == x_old, ExcInternalError());
    std::cout << std::endl;
  }
}

int
main(int argc, char* argv[])
{
  deallog.depth_console(10);
  ::dealii::MultithreadInfo::set_thread_limit((argc > 1) ? atoi(argv[1]) : 1);
  std::cout << ::dealii::MultithreadInfo::n_threads() << std::endl;
  try
  {
    run<2>(0, 0, 1);
    //  run<2>(1, 2, 2);
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
