
#include "meshworker_data.h"
#include <deal.II/fe/fe_q.h>

template <int dim>
void
run(unsigned int grid_index, unsigned int refine, unsigned int degree)
{
  FE_Q<dim> fe(degree);
  MeshworkerData<dim> data(grid_index, refine, fe);
}

int
main()
{
  deallog.depth_console(10);
  run<2>(0, 2, 2);
  run<2>(1, 2, 2);

  return 0;
}
