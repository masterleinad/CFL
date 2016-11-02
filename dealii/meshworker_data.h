
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/output.h>
#include <deal.II/meshworker/simple.h>

#include <cfl/forms.h>
#include <cfl/dealii.h>

#include <string>

using namespace dealii;
using namespace CFL::dealii::MeshWorker;

template <int dim, class FORM>
class MeshworkerIntegrator : public ::dealii::MeshWorker::LocalIntegrator<dim>
{
  const FORM& form;
  public:
  MeshworkerIntegrator(const FORM& form)
    : form(form)
    {
      this->use_boundray = false;
      this->use_faces = false;
    }
  
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
  {
  }
};


template <int dim>
class MeshworkerData
{
  MappingQ1<dim> mapping;
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
    dof.initialize_local_block_info();

    deallog << "Grid type " << grid_index << " Cells " << tr.n_active_cells() << " DoFs "
            << dof.n_dofs() << std::endl;
  }

  template <class Form>
  void
  vmult(Vector<double>& dst, const Vector<double>& src, const Form& form)
  {
    AnyData in;
    AnyData out;

    UpdateFlags update_flags =
      update_values | update_gradients | update_hessians | update_JxW_values;

    MeshWorker::IntegrationInfoBox<dim> info_box;
    // Determine degree of form and adjust
    info_box.cell_selector.add("u", true, true, true);
    info_box.boundary_selector.add("u", true, true, true);
    info_box.face_selector.add("u", true, true, true);

    info_box.add_update_flags_all(update_flags);
    info_box.initialize(dof.get_fe(), this->mapping, in, Vector<double>(), &dof.block_info());

    MeshWorker::DoFInfo<dim> dof_info(dof.block_info());

    MeshWorker::Assembler::ResidualSimple<Vector<double>> assembler;
    assembler.initialize(this->constraints());
    assembler.initialize(out);

    // Loop call
  }
};
