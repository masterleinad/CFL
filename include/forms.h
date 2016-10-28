#ifndef cfl_forms_h
#define cfl_forms_h

#include <array>
#include <iostream>
#include <string>

#include <traits.h>

namespace CFL
{
/**
 * A Form is an expression tested by a test function set.
 */
template <class Test, class Expr>
struct Form
{
  const Test test;
  const Expr expr;

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
    std::array<int, Test::Traits::dim> dummy;
    dummy.fill(0);
    std::string output;
    if (Test::Traits::rank == 0)
      output = "\\left(" + expr.latex(dummy) + "," + test.latex(dummy) + "\\right)";
    else if (Test::Traits::rank == 1)
      for (unsigned int i = 0; i < Test::Traits::dim; ++i)
      {
        if (i > 0)
          output += " + ";
        output += "\\left(" + expr.latex(dummy, i) + "," + test.latex(dummy, i) + "\\right)";
      }
    else if (Test::Traits::rank == 2)
      for (unsigned int i = 0; i < Test::Traits::dim; ++i)
        for (unsigned int j = 0; j < Test::Traits::dim; ++j)
        {
          if (i > 0 || j > 0)
            output += " + ";
          output +=
            "\\left(" + expr.latex(dummy, i, j) + "," + test.latex(dummy, i, j) + "\\right)";
        }

    return output;
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
