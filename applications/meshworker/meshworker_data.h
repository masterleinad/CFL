#ifndef _MESHWORKER_DATA_H
#define _MESHWORKER_DATA_H

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

#include <cfl/base/forms.h>
#include <cfl/meshworker/fefunctions.h>
#include <cfl/meshworker/forms.h>

#include <string>

namespace CFL::dealii::MeshWorker
{
template <int dim, class FORM>
class MeshWorkerIntegrator : public ::dealii::MeshWorker::LocalIntegrator<dim>
{
  const FORM& form;

public:
  explicit MeshWorkerIntegrator(const FORM& form)
    :  form(form)
  {
    this->use_boundary = false;
    this->use_face = false;
    // TODO(darndt): Determine from form.
    this->input_vector_names.push_back("u");
  }

  void
  cell(::dealii::MeshWorker::DoFInfo<dim>& /*dinfo*/,
       ::dealii::MeshWorker::IntegrationInfo<dim>& info) const override
  {
    anchor(form, info, *this);
    reinit(form, info);
    for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
    {
      // std::cerr << '<' << &info
      //    << ',' << info.gradients[0][0][k]
      // << '>';
      /*for (unsigned int i = 0; i < info.fe_values(0).dofs_per_cell; ++i)
        dinfo.vector(0).block(0)[i] += form.evaluate(k, i) * info.fe_values(0).JxW(k);*/
    }
  }
};

template <int dim>
class MeshworkerData
{
  ::dealii::MappingQ1<dim> mapping;
  ::dealii::SphericalManifold<dim> sphere;
  ::dealii::Triangulation<dim> tr;
  ::dealii::DoFHandler<dim> dof;

public:
  MeshworkerData(unsigned int grid_index, unsigned int refine, const ::dealii::FiniteElement<dim>& fe)
    : dof(tr)
  {
    if (grid_index == 0)
      ::dealii::GridGenerator::hyper_cube(tr);
    else if (grid_index == 1)
    {
      ::dealii::GridGenerator::hyper_ball(tr);
      tr.set_manifold(0, sphere);
      tr.set_all_manifold_ids(0);
    }
    else
      throw std::logic_error(std::string("Unknown grid index") + std::to_string(grid_index));

    tr.refine_global(refine);
    dof.distribute_dofs(fe);
    dof.initialize_local_block_info();

    ::dealii::deallog << "Grid type " << grid_index << " Cells " << tr.n_active_cells() << " DoFs "
                    << dof.n_dofs() << std::endl;
  }

  void
  resize_vector(::dealii::Vector<double>& v) const
  {
    v.reinit(dof.n_dofs());
  }

  template <class Form>
  void
  vmult(::dealii::Vector<double>& dst, const ::dealii::Vector<double>& src, Form& form) const
  {
    ::dealii::AnyData in;
    in.add<const ::dealii::Vector<double>*>(&src, "u");
    ::dealii::AnyData out;
    out.add<::dealii::Vector<double>*>(&dst, "result");

    MeshWorkerIntegrator<dim, Form> integrator(form);

    ::dealii::UpdateFlags update_flags = ::dealii::update_values | ::dealii::update_gradients |
                                       ::dealii::update_hessians | ::dealii::update_JxW_values;

    ::dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
    // Determine degree of form and adjust
    for (auto i = integrator.input_vector_names.begin(); i != integrator.input_vector_names.end();
         ++i)
    {
      // std::cerr << "dealii::Vector " << *i << std::endl;
      info_box.cell_selector.add(*i, true, true, false);
      info_box.boundary_selector.add(*i, true, true, false);
      info_box.face_selector.add(*i, true, true, false);
    }

    info_box.add_update_flags_all(update_flags);
    info_box.initialize(
      dof.get_fe(), this->mapping, in, ::dealii::Vector<double>(), &dof.block_info());

    ::dealii::MeshWorker::DoFInfo<dim> dof_info(dof.block_info());

    ::dealii::MeshWorker::Assembler::ResidualSimple<::dealii::Vector<double>> assembler;
    //    assembler.initialize(this->constraints());
    assembler.initialize(out);

    // Loop call
    ::dealii::MeshWorker::integration_loop(
      dof.begin_active(), dof.end(), dof_info, info_box, integrator, assembler);
  }
};
}

#endif // _MESHWORKER_DATA_H
