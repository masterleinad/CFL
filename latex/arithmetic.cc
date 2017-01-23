// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include <cfl/constants.h>
#include <cfl/derivatives.h>
#include <cfl/forms.h>
#include <cfl/products.h>
#include <cfl/sums.h>
#include <cfl/terminal_strings.h>
#include <iostream>

using namespace CFL;

template <class T>
void
print_form(const T& t)
{
  std::cout << "\\begin{gather*}" << std::endl
            << t.latex() << std::endl
            << "\\end{gather*}" << std::endl
            << std::endl;
}

template <class A, class B>
auto make_form(const A& a, const B& b)
{
  return Form<A, B>(a, b);
}

int
main()
{
  TerminalString<0, 2> p("p");
  TerminalString<0, 2> q("q");
  TerminalString<1, 2> u("u");

  TerminalString<0, 2, true> phi("\\phi ");

  assert_is_summable(p, q);
  print_form(make_form(phi, p + q));
  print_form(make_form(phi, p * q));
  print_form(make_form(phi, scale(4., p) * q));
  print_form(make_form(grad(phi), grad(p) + grad(q)));
  print_form(make_form(grad(grad(phi)), grad(grad(p)) + grad(grad(q))));
  print_form(make_form(grad(phi), u + grad(q)));
  print_form(make_form(grad(phi), p * grad(q)));
  print_form(make_form(grad(phi), u * p));

  return 0;
}
