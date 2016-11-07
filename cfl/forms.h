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
  std::string
  operator()(const Test& test, const Expr& expr)
  {
    return std::string("Not implemented for rank ") + std::to_string(rank);
  }
};

template <class Test, class Expr>
struct form_latex_aux<0, Test, Expr>
{
  std::string
  operator()(const Test& test, const Expr& expr)
  {
    return "\\left(" + expr.latex() + "," + test.latex() + "\\right)";
  }
};

template <class Test, class Expr>
struct form_latex_aux<1, Test, Expr>
{
  std::string
  operator()(const Test& test, const Expr& expr)
  {
    std::string output;
    for (unsigned int i = 0; i < Test::Traits::dim; ++i)
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
  std::string
  operator()(const Test& test, const Expr& expr)
  {
    std::string output;
    for (unsigned int i = 0; i < Test::Traits::dim; ++i)
      for (unsigned int j = 0; j < Test::Traits::dim; ++j)
      {
        if (i > 0 || j > 0)
          output += " + ";
        output += "\\left(" + expr.latex(i, j) + "," + test.latex(i, j) + "\\right)";
      }
    return output;
  }
};

/**
 * A Form is an expression tested by a test function set.
 */
template <class Test, class Expr>
class Form
{
 public:
  const Test test;
  const Expr expr;

  Form(const Test& test, const Expr& expr)
    : test(test)
    , expr(expr)
  {
    static_assert(Traits::is_test_function_set<Test>::value,
                  "First argument must be test function set");
    static_assert(!Traits::is_test_function_set<Expr>::value,
                  "Second argument cannot be test function set");
    static_assert(Test::Traits::rank == Expr::Traits::rank,
                  "Test functions and expression must have same tensor rank");
    static_assert(Test::Traits::dim == Expr::Traits::dim,
                  "Test functions and expression must have same dimension");
  }

  std::string
  latex() const
  {
    return form_latex_aux<Test::Traits::rank, Test, Expr>()(test, expr);
  }
};

namespace Traits
{
  template <class Test, class Expr>
  struct is_form<Form<Test, Expr>>
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
}

#endif
