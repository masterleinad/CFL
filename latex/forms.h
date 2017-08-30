#ifndef FORMS_H
#define FORMS_H

namespace CFL::Latex
{
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
transform(const Form<Test, Expr, kind_of_form, NumberType>& f)
{
  return LatexForm<Test,decltype(Latex::transform(std::declval<Expr>())), kind_of_form> (f);
}

template <typename... Types>
auto
transform(const Forms<Types...>& f)
{
  return LatexForms<decltype(transform(std::declval<Types>()))...>(f);
}
}


#endif // FORMS_H
