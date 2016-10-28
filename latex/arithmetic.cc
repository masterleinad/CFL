
#include <sums.h>
#include <iostream>
#include <terminal_strings.h>
#include <derivatives.h>
#include <forms.h>

using namespace CFL;

template <class T>
void print_form(const T& t)
{
  std::cout << "\\begin{gather*}" << std::endl
	    << t.latex() << std::endl
	    << "\\end{gather*}" << std::endl << std::endl;
}

template <class A, class B>
auto make_form(const A&a, const B&b)
{
  return Form<A,B>(a,b);
}


int main()
{
  TerminalString<0, 2> u("u");
  TerminalString<0, 2> w("w");
  TerminalString<0, 2, true> v("v");

  auto f1 = make_form(v,u+w);
  print_form(f1);
  
  auto f2 = make_form(grad(v), grad(u)+grad(w));
  print_form(f2);
  
  auto f3 = make_form(grad(grad(v)), grad(grad(u)+grad(w)));
  print_form(f3);
  
  return 0;
}


