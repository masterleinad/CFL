#ifndef cfl_forms_h
#define cfl_forms_h

#include <array>
#include <iostream>
#include <string>

#include <traits.h>

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
  const Test test;
  const Expr expr;

public:
  Form(const Test& test, const Expr& expr)
    : test(test)
    , expr(expr)
  {
    static_assert(Traits::is_test_function_set<Test>::value,
                  "First argument must be test function set");
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
}

#endif
