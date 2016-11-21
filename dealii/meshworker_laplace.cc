
#include "meshworker_data.h"
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>

#include <cfl/cfl.h>
#include <cfl/dealii.h>

using namespace CFL;

template <int dim>
void
run(unsigned int grid_index, unsigned int refine, unsigned int degree)
{
  FE_Q<dim> fe(degree);
  MeshworkerData<dim> data(grid_index, refine, fe);

  ScalarTestFunction<dim> v(0);
  FEFunction<0, dim> u("u", 0);
  auto Dv = grad(v);
  auto Du = grad(u);
  auto f = form(Du, Dv);

  Vector<double> x;
  Vector<double> b;
  data.resize_vector(x);
  data.resize_vector(b);

  for (unsigned int i = 0; i < b.size(); ++i)
  {
    if (i)
      b[i - 1] = 0.;
    b[i] = 1.;
    x = 0.;
    data.vmult(x, b, f);
    for (unsigned int j = 0; j < x.size(); ++j)
      if (x[j] != 0.)
        std::cout << i << '\t' << j << '\t' << x[j] << std::endl;
    std::cout << std::endl;
  }
}

int
main()
{
  deallog.depth_console(10);
  run<2>(0, 2, 2);
  //  run<2>(1, 2, 2);

  return 0;
}
