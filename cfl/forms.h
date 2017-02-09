#ifndef cfl_forms_h
#define cfl_forms_h

#include <array>
#include <iostream>
#include <string>

#include <cfl/traits.h>

namespace CFL
{
template <int rank, class Test, class Expr>
struct form_latex_aux
{
  std::string operator()(const Test& test, const Expr& expr)
  {
    return std::string("Not implemented for rank ") + std::to_string(rank);
  }
};

template <class Test, class Expr>
struct form_latex_aux<0, Test, Expr>
{
  std::string operator()(const Test& test, const Expr& expr)
  {
    return "\\left(" + expr.latex() + "," + test.latex() + "\\right)";
  }
};

template <class Test, class Expr>
struct form_latex_aux<1, Test, Expr>
{
  std::string operator()(const Test& test, const Expr& expr)
  {
    std::string output;
    for (unsigned int i = 0; i < Test::TensorTraits::dim; ++i)
    {
      if (i > 0)
        output += " + ";
      output += "\\left(" + expr.latex(i) + "," + test.latex(i) + "\\right)";
    }
    return output;
  }
};

template <class Test, class Expr>
struct form_latex_aux<2, Test, Expr>
{
  std::string operator()(const Test& test, const Expr& expr)
  {
    std::string output;
    for (unsigned int i = 0; i < Test::TensorTraits::dim; ++i)
      for (unsigned int j = 0; j < Test::TensorTraits::dim; ++j)
      {
        if (i > 0 || j > 0)
          output += " + ";
        output += "\\left(" + expr.latex(i, j) + "," + test.latex(i, j) + "\\right)";
      }
    return output;
  }
};

template <int rank, class Test, class Expr>
struct form_evaluate_aux
{
  double operator()(unsigned int k, unsigned int i, const Test& test, const Expr& expr)
  {
    static_assert(rank < 2, "Not implemented for this rank");
    return 0.;
  }
};

template <class Test, class Expr>
struct form_evaluate_aux<0, Test, Expr>
{
  double operator()(unsigned int k, unsigned int i, const Test& test, const Expr& expr)
  {
    return test.evaluate(k, i) * expr.evaluate(k);
  }
};

template <class Test, class Expr>
struct form_evaluate_aux<1, Test, Expr>
{
  double operator()(unsigned int k, unsigned int i, const Test& test, const Expr& expr)
  {
    double sum = 0.;
    for (unsigned int d = 0; d < Test::TensorTraits::dim; ++d)
      sum += test.evaluate(k, i, d) * expr.evaluate(k, d);
    return sum;
  }
};

template <typename... Types>
class Forms;

/**
 * A Form is an expression tested by a test function set.
 */
template <class Test, class Expr, typename number = double>
class Form
{
public:
  const Test test;
  const Expr expr;

  static constexpr unsigned int fe_number = Test::index;
  static constexpr bool integrate_value = Test::integrate_value;
  static constexpr bool integrate_gradient = Test::integrate_gradient;

  Form(const Test& test, const Expr& expr)
    : test(test)
    , expr(expr)
  {
    std::cout << "constructor1" << std::endl;
    static_assert(Traits::is_test_function_set<Test>::value,
                  "First argument must be test function set");
    static_assert(!Traits::is_test_function_set<Expr>::value,
                  "Second argument cannot be test function set");
    static_assert(Test::TensorTraits::rank == Expr::TensorTraits::rank,
                  "Test functions and expression must have same tensor rank");
    static_assert(Test::TensorTraits::dim == Expr::TensorTraits::dim,
                  "Test functions and expression must have same dimension");
  }

  std::string
  latex() const
  {
    return form_latex_aux<Test::TensorTraits::rank, Test, Expr>()(test, expr);
  }

  template <class FEEvaluation>
  static void
  integrate(FEEvaluation& phi)
  {
    // only to be used if there is only one form!
    phi.template integrate<fe_number>(integrate_value, integrate_gradient);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags(FEEvaluation& phi)
  {
    // only to be used if there is only one form!
    phi.template set_integration_flags<fe_number>(integrate_value, integrate_gradient);
  }

  template <class FEEvaluation>
  void
  set_evaluation_flags(FEEvaluation& phi) const
  {
    // only to be used if there is only one form!
    expr.set_evaluation_flags(phi);
  }

  number
  evaluate(unsigned int k, unsigned int i) const
  {
    return form_evaluate_aux<Test::TensorTraits::rank, Test, Expr>()(k, i, test, expr);
  }

  template <class FEEvaluation>
  void
  evaluate(FEEvaluation& phi, unsigned int q) const
  {
    // only to be used if there is only one form!
    const auto value = expr.value(phi, q);
    Test::submit(phi, q, value);
  }

  template <class FEEvaluation>
  auto value(FEEvaluation& phi, unsigned int q) const
  {
    return expr.value(phi, q);
  }

  template <class FEEvaluation, typename ValueType>
  static void
  submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
  {
    Test::submit(phi, q, value);
  }

  template <class TestNew, class ExprNew>
  Forms<Form<Test, Expr>, Form<TestNew, ExprNew>> operator+(
    const Form<TestNew, ExprNew>& new_form) const
  {
    std::cout << "operator+1" << std::endl;
    return Forms<Form<Test, Expr>, Form<TestNew, ExprNew>>(*this, new_form);
  }

  template <class... Types>
  auto operator+(const Forms<Types...>& old_form) const
  {
    return old_form + *this;
  }

  Form<Test, Expr, number> operator-() const
  {
    const typename std::remove_reference<decltype(*this)>::type newform(test, -expr);
    return newform;
  }
};

namespace Traits
{
  template <class Test, class Expr>
  struct is_form<Form<Test, Expr>>
  {
    const static bool value = true;
  };

  template <class Test, class Expr>
  struct is_cfl_object<Form<Test, Expr>>
  {
    const static bool value = true;
  };

  template <class Test1, class Expr1, class Test2, class Expr2>
  struct is_summable<Form<Test1, Expr1>, Form<Test2, Expr2>>
  {
    const static bool value = true;
  };

  template <class Test, class Expr, typename... Types>
  struct is_summable<Form<Test, Expr>, Forms<Types...>>
  {
    const static bool value = true;
  };

  template <class Test, class Expr, typename... Types>
  struct is_summable<Forms<Types...>, Form<Test, Expr>>
  {
    const static bool value = true;
  };
}

template <class Test, class Expr>
typename std::enable_if<Traits::is_test_function_set<Test>::value, Form<Test, Expr>>::type
form(const Test& t, const Expr& e)
{
  return Form<Test, Expr>(t, e);
}

template <class Test, class Expr>
typename std::enable_if<Traits::is_test_function_set<Test>::value, Form<Test, Expr>>::type
form(const Expr& e, const Test& t)
{
  return Form<Test, Expr>(t, e);
}

template <typename... Types>
class Forms
{
public:
  Forms() = delete;
  Forms(const Forms<Types...>&) = delete;
};

template <typename FormType>
class Forms<FormType>
{
public:
  static constexpr bool integrate_value = FormType::integrate_value;
  static constexpr bool integrate_gradient = FormType::integrate_gradient;
  static constexpr unsigned int fe_number = FormType::fe_number;

  Forms(const FormType& form)
    : form(form)
  {
    std::cout << "constructor2" << std::endl;
    static_assert(Traits::is_form<FormType>::value,
                  "You need to construct this with a Form object!");
  }

  template <class FEEvaluation>
  static void
  set_integration_flags(FEEvaluation& phi)
  {
    phi.template set_integration_flags<fe_number>(integrate_value, integrate_gradient);
  }

  template <class FEEvaluation>
  void
  set_evaluation_flags(FEEvaluation& phi) const
  {
    form.expr.set_evaluation_flags(phi);
  }

  template <class FEEvaluation>
  void
  evaluate(FEEvaluation& phi, unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "expecting value " << fe_number << std::endl;
#endif
    const auto value = form.value(phi, q);
#ifdef DEBUG_OUUTPUT
    std::cout << "expecting submit " << fe_number << std::endl;
#endif
    form.submit(phi, q, value);
  }

  template <class FEEvaluation>
  static void
  integrate(FEEvaluation& phi)
  {
    phi.template integrate<fe_number>(integrate_value, integrate_gradient);
  }

private:
  const FormType& form;
};

template <typename FormType, typename... Types>
class Forms<FormType, Types...> : public Forms<Types...>
{
public:
  static constexpr bool integrate_value = FormType::integrate_value;
  static constexpr bool integrate_gradient = FormType::integrate_gradient;
  static constexpr unsigned int fe_number = FormType::fe_number;

  Forms(const FormType& form, const Forms<Types...>& old_form)
    : Forms<Types...>(old_form)
    , form(form)
  {
    std::cout << "constructor3" << std::endl;
    static_assert(Traits::is_form<FormType>::value,
                  "You need to construct this with a Form object!");
  }

  Forms(const FormType& form, const Types&... old_form)
    : Forms<Types...>(old_form...)
    , form(form)
  {
    std::cout << "constructor4" << std::endl;
    static_assert(Traits::is_form<FormType>::value,
                  "You need to construct this with a Form object!");
  }

  template <class Test, class Expr>
  Forms<Form<Test, Expr>, FormType, Types...> operator+(const Form<Test, Expr>& new_form) const
  {
    std::cout << "operator+2" << std::endl;
    return Forms<Form<Test, Expr>, FormType, Types...>(new_form, *this);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags(FEEvaluation& phi)
  {
    phi.template set_integration_flags<fe_number>(integrate_value, integrate_gradient);
    Forms<Types...>::set_integration_flags(phi);
  }

  template <class FEEvaluation>
  void
  set_evaluation_flags(FEEvaluation& phi) const
  {
    form.expr.set_evaluation_flags(phi);
    Forms<Types...>::set_evaluation_flags(phi);
  }

  template <class FEEvaluation>
  void
  evaluate(FEEvaluation& phi, unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "expecting value " << fe_number << std::endl;
#endif
    const auto value = form.value(phi, q);
#ifdef DEBUG_OUTPUT
    std::cout << "descending" << std::endl;
#endif
    Forms<Types...>::evaluate(phi, q);
#ifdef DEBUG_OUTPUT
    std::cout << "expecting submit " << fe_number << std::endl;
#endif
    form.submit(phi, q, value);
  }

  template <class FEEvaluation>
  static void
  integrate(FEEvaluation& phi)
  {
    phi.template integrate<fe_number>(integrate_value, integrate_gradient);
    Forms<Types...>::integrate(phi);
  }

private:
  const FormType& form;
};
}

#endif
