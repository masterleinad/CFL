
#include <cfl/derivatives.h>
#include <cfl/forms.h>
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

int
main()
{
  TerminalString<0, 2> p("p");
  TerminalString<0, 2, true> q("q");
  TerminalString<1, 2> u("u");
  TerminalString<1, 2, true> v("v");

  print_form(form(q, p));
  print_form(form(p, q));
  print_form(form(v, u));
  print_form(form(grad(q), u));
  print_form(form(v, grad(p)));
  print_form(form(grad(v), grad(u)));
  print_form(form(grad(grad(q)), grad(u)));
  print_form(form(grad(v), grad(grad(p))));
  print_form(form(grad(grad(q)), grad(grad(p))));

  return 0;
}
