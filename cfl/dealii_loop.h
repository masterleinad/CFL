#ifndef CFL_DEALII_LOOP_H
#define CFL_DEALII_LOOP_H

namespace CFL
{
namespace dealii
{
  namespace MeshWorker
  {
    template <int dim, class Operator>
    class Engine
    {
      Operator op;
      std::vector<TestFunctionIdentifier> unique_test_function_spaces;

      template <int dim, class Expr, class Test>
      std::enable_if<Test::rank == 0, void>::type
      integrate(Test& test, const Expr& expr, const IntegrationInfo<dim>& info)
      {
        static_assert(Expr::rank == Test::rank,
                      "Expression and test function must have equal rank");
        // Loop over all quadrature points
        for (unsigned int q = 0; q < info.fe_values(0).n_quadrature_points; ++q)
        {
          const double e = expr.value(q) * fe_values(0).JxW(q);
          for (unsigned int i = 0; i < ; ++i)
            test.store(i, e * test.value(i, q));
        }
      }

      template <int dim, class Expr, class Test>
      std::enable_if<Test::rank == 1, void>::type
      integrate(Test& test, const Expr& expr, const IntegrationInfo<dim>& info)
      {
        static_assert(Expr::rank == Test::rank,
                      "Expression and test function must have equal rank");
        // Loop over all quadrature points
        for (unsigned int q = 0; q < info.fe_values(0).n_quadrature_points; ++q)
          for (unsigned int d = 0; d < dim; ++d)
          {
            const double e = expr.value(d, q) * fe_values(0).JxW(q);
            for (unsigned int i = 0; i < ; ++i)
              test.store(i, e * test.value(i, d, q));
          }
      }

      template <int dim, class Expr, class Test>
      std::enable_if<Test::rank == 2, void>::type
      integrate(Test& test, const Expr& expr, const IntegrationInfo<dim>& info)
      {
        static_assert(Expr::rank == Test::rank,
                      "Expression and test function must have equal rank");
        // Loop over all quadrature points
        for (unsigned int q = 0; q < info.fe_values(0).n_quadrature_points; ++q)
          for (unsigned int d1 = 0; d1 < dim; ++d1)
            for (unsigned int d2 = 0; d2 < dim; ++d2)
            {
              const double e = expr.value(d1, d2, q) * fe_values(0).JxW(q);
              for (unsigned int i = 0; i < ; ++i)
                test.store(i, e * test.value(i, d1, d2, q));
            }
      }

      void
      cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
      {
        // Initialize all finite element function objects such that
        // they access the local data vectors
      }
      void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
      void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                IntegrationInfo<dim>& info2) const;

    public:
      void
      operator()(dealii::AnyData& out, const dealii::AnyData& in)
      {
        UpdateFlags update_flags = integrator.update_flags();
        bool values_flag = update_flags & update_values;
        bool gradients_flag = update_flags & update_gradients;
        bool hessians_flag = update_flags & update_hessians;
      }
    };
  }
}
}

#endif
