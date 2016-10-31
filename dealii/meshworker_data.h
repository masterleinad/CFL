
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <string>

using namespace dealii;

template <int dim>
class MeshworkerData
{
  SphericalManifold<dim> sphere;
  Triangulation<dim> tr;
  DoFHandler<dim> dof;

public:
  MeshworkerData(unsigned int grid_index, unsigned int refine, const FiniteElement<dim>& fe)
    : dof(tr)
  {
    if (grid_index == 0)
      GridGenerator::hyper_cube(tr);
    else if (grid_index == 1)
    {
      GridGenerator::hyper_ball(tr);
      tr.set_manifold(0, sphere);
      tr.set_all_manifold_ids(0);
    }
    else
      throw std::logic_error(std::string("Unknown grid index") + std::to_string(grid_index));

    tr.refine_global(refine);
    dof.distribute_dofs(fe);

    deallog << "Grid type " << grid_index << " Cells " << tr.n_active_cells() << " DoFs "
            << dof.n_dofs() << std::endl;
  }
};
