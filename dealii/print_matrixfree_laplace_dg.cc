// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#define DEBUG_OUTPUT

#include "matrixfree_data.h"
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <cfl/cfl.h>
#include <cfl/dealii_matrixfree.h>
#include <cfl/forms.h>

// To generate a reference solution
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include <fstream>

#include <latex/fefunctions.h>

using namespace CFL;

template <class LatexTest, class LatexExpr, FormKind kind_of_form>
class LatexForm
{
public:

    template <class Test, class Expr, typename NumberType>
    LatexForm(const Form<Test, Expr, kind_of_form, NumberType> f):
    expr(Latex::transform(f.expr)), test(f.test)
    {}

  std::string
  print(const std::vector<std::string>& function_names,
        const std::vector<std::string>& expression_names) const
  {
    const std::string domain = []() {
      switch (kind_of_form)
      {
        case FormKind::cell:
          return R"(_\Omega)";
          break;
        case FormKind::face:
          return R"(_F)";
          break;
        case FormKind::boundary:
          return R"(_{\partial \Omega})";
          break;
        default:
          Assert(false, ::dealii::ExcInternalError());
      }
    }();
    return "(" + expr.value(function_names) + "," + test.print(expression_names) + ")" + domain;
  }

private:
  const LatexExpr expr;
  const LatexTest test;
};

template <typename... Types>
class LatexForms;

template <typename FormType, typename... FormTypes>
class LatexForms<FormType, FormTypes...> : public LatexForms<FormTypes...>
{
public:
  template <class OtherType, class... OtherTypes,
            typename std::enable_if<sizeof...(OtherTypes)==sizeof...(FormTypes)>::type* = nullptr>
  LatexForms(const Forms<OtherType, OtherTypes...>&f)
    :LatexForms<FormTypes...>(static_cast<Forms<OtherTypes...>>(f)),
     form(f.get_form())
  {}

  std::string
  print(const std::vector<std::string>& function_names,
        const std::vector<std::string>& expression_names) const
  {
    return form.print(function_names, expression_names) + "+" +
           LatexForms<FormTypes...>::print(function_names, expression_names);
  }

private:
  const FormType form;
};

template <class Test, class Expr, FormKind kind_of_form>
class LatexForms<LatexForm<Test, Expr, kind_of_form>>
{
public:  
template<class OtherExpr, typename NumberType>
    LatexForms(const Forms<Form<Test, OtherExpr, kind_of_form, NumberType>>&f):form(f.get_form()){}

  template <typename NumberType>
  void initialize (const Form<Test, Expr, kind_of_form, NumberType>) {}

  std::string
  print(const std::vector<std::string>& function_names,
        const std::vector<std::string>& expression_names) const
  {
    return form.print(function_names, expression_names);
  }

private:
  const LatexForm<Test, Expr, kind_of_form> form;
};

template <class Test, class Expr, FormKind kind_of_form, typename NumberType>
auto
transform_latex(const Form<Test, Expr, kind_of_form, NumberType>& f)
{
  return LatexForm<Test,decltype(Latex::transform(std::declval<Expr>())), kind_of_form> (f);
}

template <typename... Types>
auto
transform_latex(const Forms<Types...>& f)
{
  return LatexForms<decltype(transform_latex(std::declval<Types>()))...>(f);
}


template<class FormContainer>
class LatexEvaluator
{
public:
  LatexEvaluator (const FormContainer& form_container,
                  const std::vector<std::string> & function_names,
                  const std::vector<std::string> & test_names)
    : _form_container(form_container),
      _function_names(function_names),
      _test_names(test_names)
  {}

  void print()
  {
    std::cout << _form_container.print(_function_names, _test_names) << std::endl;
  }

private:
  const FormContainer& _form_container;
  const std::vector<std::string>& _function_names;
  const std::vector<std::string>& _test_names;
};


void
test()
{
  CFL::dealii::MatrixFree::TestFunction<0, 1, 0> v;
  auto Dv = grad(v);
  CFL::dealii::MatrixFree::TestFunctionInteriorFace<0, 1, 0> v_p;
  CFL::dealii::MatrixFree::TestFunctionExteriorFace<0, 1, 0> v_m;
  CFL::dealii::MatrixFree::TestNormalGradientInteriorFace<0, 1, 0> Dnv_p;
  CFL::dealii::MatrixFree::TestNormalGradientExteriorFace<0, 1, 0> Dnv_m;

  CFL::dealii::MatrixFree::FEFunction<0, 1, 0> u("u");
  auto Du = grad(u);
  CFL::dealii::MatrixFree::FEFunctionInteriorFace<0, 1, 0> u_p("u+");
  CFL::dealii::MatrixFree::FEFunctionExteriorFace<0, 1, 0> u_m("u-");
  CFL::dealii::MatrixFree::FENormalGradientInteriorFace<0, 1, 0> Dnu_p("u+");
  CFL::dealii::MatrixFree::FENormalGradientExteriorFace<0, 1, 0> Dnu_m("u-");

  auto cell = form(Du, Dv);

  auto flux = u_p - u_m;
  auto flux_grad = Dnu_p - Dnu_m;

  auto flux1 = -face_form(flux, Dnv_p) + face_form(flux, Dnv_m);
  auto flux2 = face_form(-flux + .5 * flux_grad, v_p) - face_form(-flux + .5 * flux_grad, v_m);

  auto boundary1 = boundary_form(2. * u_p - Dnu_p, v_p);
  auto boundary3 = -boundary_form(u_p, Dnv_p);

  auto face = -flux2 + .5 * flux1;
  auto f = cell + face + boundary1 + boundary3;

  std::vector<std::string> function_names(1, "u");
  std::vector<std::string> test_names(1, "v");
  const auto latex_forms = transform_latex(f);
  LatexEvaluator<decltype(latex_forms)> evaluator(latex_forms, function_names, test_names);
  evaluator.print();
}

int
main(int /*argc*/, char** /*argv*/)
{
  ::dealii::deallog.depth_console(10);
  try
  {    
    test();
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
