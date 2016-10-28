
#include <derivatives.h>
#include <forms.h>
#include <iostream>
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
  TerminalString<0, 2, true> q("q");
  TerminalString<1, 2> u("u");
  TerminalString<1, 2, true> v("v");

  print_form(make_form(q, p));
  print_form(make_form(v, u));
  print_form(make_form(grad(q), u));
  print_form(make_form(v, grad(p)));
  print_form(make_form(grad(v), grad(u)));
  print_form(make_form(grad(grad(q)), grad(u)));
  print_form(make_form(grad(v), grad(grad(p))));
  print_form(make_form(grad(grad(q)), grad(grad(p))));

  return 0;
}
