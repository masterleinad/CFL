// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include "matrixfree_data.h"
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/vector.h>

#include <cfl/cfl.h>
#include <cfl/dealii_matrixfree.h>

using namespace CFL;

template <int dim>
void
run(unsigned int grid_index, unsigned int refine, unsigned int degree)
{
  FE_Q<dim> fe_p(degree);
  FEData</*degree*/ 1, /*components*/ 1, dim, /*fe_no*/ 0, /*max_degree*/ 1> fedata(fe_p);
  // FEDatas<decltype(fedata)> fe_datas = fedata;

  // std::vector<FiniteElement<dim>* > fes;
  // fes.push_back(&fe_p);
  // MatrixFreeData<dim, decltype(fe_datas) > data(grid_index, refine, fes, fe_datas);
  auto data = make_matrix_free_data(grid_index, refine, fe_p, fedata);
  data.initialize();

  TestFunction<0, dim, 0> q;
  FEFunction<0, dim, 0> p("p");

  auto f1 = form(p, q);
  auto f2 = form(-p, q);
  auto f3 = -f2;

  LinearAlgebra::distributed::BlockVector<double> in(1);
  LinearAlgebra::distributed::BlockVector<double> out1(1), out2(1), out3(1);
  std::cout << in.n_blocks() << std::endl;
  data.resize_vector(in);
  data.resize_vector(out1);
  data.resize_vector(out2);
  data.resize_vector(out3);

  for (size_t i = 0; i < in.n_blocks(); ++i)
  {
    for (types::global_dof_index j = 0; j < in.block(i).size(); ++j)
      in.block(i)[j] = j;
  }
  {
    data.vmult(out1, in, f1);
    data.vmult(out2, in, f2);
    data.vmult(out3, in, f3);
    for (size_t k = 0; k < in.n_blocks(); ++k)
    {
      for (types::global_dof_index j = 0; j < in.block(k).size(); ++j)
      {
        std::cout << j << '\t' << k << '\t' << out1.block(k)[j] << '\t' << out2.block(k)[j] << '\t'
                  << out3.block(k)[j] << std::endl;
      }
    }

    for (size_t k = 0; k < in.n_blocks(); ++k)
    {
      out1.block(k) += out2.block(k);
      std::cout << k << " error: " << out1.block(k).l2_norm() << std::endl;
      Assert(out1.block(k).l2_norm() < 1.e-20 ||
               out1.block(k).l2_norm() < 1.e-6 * out2.block(k).l2_norm(),
             ExcInternalError());
      out3.block(k) += out2.block(k);
      std::cout << k << " error: " << out3.block(k).l2_norm() << std::endl;
      Assert(out3.block(k).l2_norm() < 1.e-20 ||
               out3.block(k).l2_norm() < 1.e-6 * out2.block(k).l2_norm(),
             ExcInternalError());
    }
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
