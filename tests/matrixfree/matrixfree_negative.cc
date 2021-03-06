// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#define DEBUG_OUTPUT

#include "matrixfree_data.h"
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/vector.h>

#include <cfl/matrixfree/fefunctions.h>
#include <cfl/matrixfree/forms.h>

using namespace dealii;
using namespace CFL;
using namespace CFL::dealii::MatrixFree;

template <int dim>
void
run(unsigned int grid_index, unsigned int refine, unsigned int degree)
{
  FE_Q<dim> fe_p(degree);
  FEData<FE_Q, /*degree*/ 1, /*components*/ 1, dim, /*fe_no*/ 0, /*max_degree*/ 1> fedata(fe_p);
  FEDatas<decltype(fedata)> fe_datas = FEDatas<decltype(fedata)>(fedata);

  std::vector<FiniteElement<dim>*> fes;
  fes.push_back(&fe_p);

  Base::TestFunction<0, dim, 0> q;
  Base::FEFunction<0, dim, 0> p;

  auto base_f1 = Base::form(p, q);
  auto base_f2 = Base::form(-p, q);
  auto base_f3 = -base_f2;

  auto f1 = transform(base_f1);
  auto f2 = transform(base_f2);
  auto f3 = transform(base_f3);

  MatrixFreeData<dim,
                 decltype(fe_datas),
                 decltype(f1),
                 LinearAlgebra::distributed::BlockVector<double>>
    data1(grid_index, refine, fes, fe_datas, f1);
  MatrixFreeData<dim,
                 decltype(fe_datas),
                 decltype(f2),
                 LinearAlgebra::distributed::BlockVector<double>>
    data2(grid_index, refine, fes, fe_datas, f2);
  MatrixFreeData<dim,
                 decltype(fe_datas),
                 decltype(f3),
                 LinearAlgebra::distributed::BlockVector<double>>
    data3(grid_index, refine, fes, fe_datas, f3);

  LinearAlgebra::distributed::BlockVector<double> in(1);
  LinearAlgebra::distributed::BlockVector<double> out1(1), out2(1), out3(1);
  std::cout << in.n_blocks() << std::endl;
  data1.resize_vector(in);
  data1.resize_vector(out1);
  data2.resize_vector(out2);
  data3.resize_vector(out3);

  for (size_t i = 0; i < in.n_blocks(); ++i)
  {
    for (types::global_dof_index j = 0; j < in.block(i).size(); ++j)
      in.block(i)[j] = j;
  }
  {
    data1.vmult(out1, in);
    data2.vmult(out2, in);
    data3.vmult(out3, in);
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
