
#include <constants.h>
#include <derivatives.h>
#include <forms.h>
#include <iostream>
#include <products.h>
#include <sums.h>
#include <terminal_strings.h>

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
auto
make_form(const A& a, const B& b)
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
