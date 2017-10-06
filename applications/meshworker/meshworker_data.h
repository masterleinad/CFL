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

#include <cfl/meshworker/fefunctions.h>
#include <cfl/meshworker/forms.h>

#include <string>

using namespace dealii;
using namespace ::dealii::MeshWorker;

namespace CFL
{
namespace dealii
{
  namespace MeshWorker
  {
    template <int dim, class Expr, class Test>
    typename std::enable_if<Test::TensorTraits::rank == 0, void>::type
    evaluate(DoFInfo<dim, dim>& dinfo, const IntegrationInfo<dim>& info, const Test& test,
             const Expr& expr, unsigned int q)
    {
      static_assert(Expr::TensorTraits::rank == Test::TensorTraits::rank,
                    "Expression and test function must have equal rank");
      const unsigned int test_fev = test.id().fe_index;
      const double e = expr.value(info, q) * info.fe_values(0).JxW(q);
      for (unsigned int i = 0; i < info.fe_values(test_fev).dofs_per_cell; ++i)
        dinfo.vector(0).block(test.id().block_index)[i] += e * test.value(i, info, q);
    }

    template <int dim, class Expr, class Test>
    typename std::enable_if<Test::TensorTraits::rank == 1, void>::type
    evaluate(DoFInfo<dim, dim>& dinfo, const IntegrationInfo<dim>& info, const Test& test,
             const Expr& expr, unsigned int q)
    {
      static_assert(Expr::TensorTraits::rank == Test::TensorTraits::rank,
                    "Expression and test function must have equal rank");
      const unsigned int test_fev = test.id().fe_index;
      for (unsigned int d = 0; d < dim; ++d)
      {
        const double e = expr.value(d, info, q) * info.fe_values(0).JxW(q);
        for (unsigned int i = 0; i < info.fe_values(test_fev).dofs_per_cell; ++i)
          dinfo.vector(0).block(test.id().block_index)[i] += e * test.value(d, i, info, q);
      }
    }

    template <int dim, class Expr, class Test>
    typename std::enable_if<Test::TensorTraits::rank == 2, void>::type
    evaluate(DoFInfo<dim, dim>& dinfo, const IntegrationInfo<dim>& info, const Test& test,
             const Expr& expr, unsigned int q)
    {
      static_assert(Expr::TensorTraits::rank == Test::TensorTraits::rank,
                    "Expression and test function must have equal rank");
      const unsigned int test_fev = test.id().fe_index;
      for (unsigned int d1 = 0; d1 < dim; ++d1)
        for (unsigned int d2 = 0; d2 < dim; ++d2)
        {
          const double e = expr.value(d1, d2, info, q) * info.fe_values(0).JxW(q);
          for (unsigned int i = 0; i < info.fe_values(test_fev).dofs_per_cell; ++i)
            dinfo.vector(0).block(test.id().block_index)[i] += e * test.value(d1, d2, i, info, q);
        }
    }

    template <FormKind kind, int dim, class FORMS>
    typename std::enable_if<FORMS::number == 0, void>::type
    evaluate(DoFInfo<dim, dim>& dinfo, const IntegrationInfo<dim>& info, const FORMS& form,
             unsigned int q)
    {
      if
        constexpr(FORMS::form_kind == kind)
          evaluate(dinfo, info, form.get_form().test(), form.get_form().expr(), q);
    }

    template <FormKind kind, int dim, class FORMS>
    typename std::enable_if<FORMS::number != 0, void>::type
    evaluate(DoFInfo<dim, dim>& dinfo, const IntegrationInfo<dim>& info, const FORMS& form,
             unsigned int q)
    {
      evaluate<kind>(dinfo, info, form.get_other(), q);
      if
        constexpr(FORMS::form_kind == kind)
          evaluate(dinfo, info, form.get_form().test(), form.get_form().expr(), q);
    }

    template <FormKind kind, int dim, class FORMS>
    typename std::enable_if<FORMS::number == 0, void>::type
    evaluate(DoFInfo<dim, dim>& dinfo1, DoFInfo<dim, dim>& dinfo2,
             const IntegrationInfo<dim>& info1, const IntegrationInfo<dim>& info2,
             const FORMS& form, unsigned int q)
    {
      if
        constexpr(FORMS::form_kind == kind)
          evaluate(dinfo1, dinfo2, info1, info2, form.get_form().test(), form.get_form().expr(), q);
    }

    template <FormKind kind, int dim, class FORMS>
    typename std::enable_if<FORMS::number != 0, void>::type
    evaluate(DoFInfo<dim, dim>& dinfo1, DoFInfo<dim, dim>& dinfo2,
             const IntegrationInfo<dim>& info1, const IntegrationInfo<dim>& info2,
             const FORMS& form, unsigned int q)
    {
      evaluate<kind>(dinfo1, dinfo2, info1, info2, form.get_other(), q);
      if
        constexpr(FORMS::form_kind == kind)
          evaluate(dinfo1, dinfo2, info1, info2, form.get_form().test(), form.get_form().expr(), q);
    }

    template <int dim, class FORM>
    class MeshWorkerIntegrator : public LocalIntegrator<dim>
    {
      const FORM& form;

    public:
      explicit MeshWorkerIntegrator(const FORM& form)
        : form(form)
      {
        this->use_boundary = false;
        this->use_face = false;
        // TODO(darndt): Determine from form.
        this->input_vector_names.push_back("u");
      }

      void
      cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override
      {
        for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
        {
          evaluate<FormKind::cell>(dinfo, info, form, k);
        }
      }

      void
      boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override
      {
        for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
        {
          evaluate<FormKind::boundary>(dinfo, info, form, k);
        }
      }

      void
      face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
           IntegrationInfo<dim>& info2) const override
      {
        for (unsigned int k = 0; k < info1.fe_values(0).n_quadrature_points; ++k)
        {
          evaluate<FormKind::face>(dinfo1, dinfo2, info1, info2, form, k);
        }
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

      void
      resize_vector(Vector<double>& v) const
      {
        v.reinit(dof.n_dofs());
      }

      template <class Form>
      void
      vmult(Vector<double>& dst, const Vector<double>& src, Form& form) const
      {
        AnyData in;
        in.add<const Vector<double>*>(&src, "u");
        AnyData out;
        out.add<Vector<double>*>(&dst, "result");

        MeshWorkerIntegrator<dim, Form> integrator(form);

        UpdateFlags update_flags =
          update_values | update_gradients | update_hessians | update_JxW_values;

        IntegrationInfoBox<dim> info_box;
        // Determine degree of form and adjust
        for (auto i = integrator.input_vector_names.begin();
             i != integrator.input_vector_names.end();
             ++i)
        {
          // std::cerr << "Vector " << *i << std::endl;
          info_box.cell_selector.add(*i, true, true, true);
          info_box.boundary_selector.add(*i, true, true, true);
          info_box.face_selector.add(*i, true, true, true);
        }

        info_box.add_update_flags_all(update_flags);
        info_box.initialize(dof.get_fe(), this->mapping, in, Vector<double>(), &dof.block_info());

        DoFInfo<dim> dof_info(dof.block_info());

        Assembler::ResidualSimple<Vector<double>> assembler;
        //    assembler.initialize(this->constraints());
        assembler.initialize(out);

        // Loop call
        integration_loop(dof.begin_active(), dof.end(), dof_info, info_box, integrator, assembler);
      }
    };
  }
}
}

#endif // _MESHWORKER_DATA_H
