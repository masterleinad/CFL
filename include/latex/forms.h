#ifndef LATEX_FORMS_H
#define LATEX_FORMS_H

#include <cfl/forms.h>
#include <latex/fefunctions.h>

#include <vector>

namespace CFL::Latex
{
template <class LatexTest, class LatexExpr, FormKind kind_of_form>
class Form
{
public:
  template <class Test, class Expr, typename NumberType>
  Form(const Base::Form<Test, Expr, kind_of_form, NumberType> f)
    : expr(transform(f.expr))
    , test(transform(f.test))
  {
  }

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
    return "(" + expr.value(function_names) + "," + test.submit(expression_names) + ")" + domain;
  }

private:
  const LatexExpr expr;
  const LatexTest test;
};

template <typename... Types>
class Forms;

template <typename FormType, typename... FormTypes>
class Forms<FormType, FormTypes...> : public Forms<FormTypes...>
{
public:
  template <class OtherType, class... OtherTypes,
            typename std::enable_if<sizeof...(OtherTypes) == sizeof...(FormTypes)>::type* = nullptr>
  Forms(const Base::Forms<OtherType, OtherTypes...>& f)
    : Forms<FormTypes...>(static_cast<Base::Forms<OtherTypes...>>(f))
    , form(f.get_form())
  {
  }

  std::string
  print(const std::vector<std::string>& function_names,
        const std::vector<std::string>& expression_names) const
  {
    return form.print(function_names, expression_names) + "+" +
           Forms<FormTypes...>::print(function_names, expression_names);
  }

private:
  const FormType form;
};

template <class Test, class Expr, FormKind kind_of_form>
class Forms<Form<Test, Expr, kind_of_form>>
{
public:
  template <class OtherTest, class OtherExpr, typename NumberType>
  Forms(const Base::Forms<Base::Form<OtherTest, OtherExpr, kind_of_form, NumberType>>& f)
    : form(f.get_form())
  {
  }

  std::string
  print(const std::vector<std::string>& function_names,
        const std::vector<std::string>& expression_names) const
  {
    return form.print(function_names, expression_names);
  }

private:
  const Form<Test, Expr, kind_of_form> form;
};

template <class Test, class Expr, FormKind kind_of_form, typename NumberType>
auto
transform(const Base::Form<Test, Expr, kind_of_form, NumberType>& f)
{
  return Form<decltype(transform(std::declval<Test>())),
              decltype(transform(std::declval<Expr>())),
              kind_of_form>(f);
}

template <typename... Types>
auto
transform(const Base::Forms<Types...>& f)
{
  return Forms<decltype(transform(std::declval<Types>()))...>(f);
}
}

#endif // LATEX_FORMS_H
