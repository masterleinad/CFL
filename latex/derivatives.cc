
#include <derivatives.h>
#include <iostream>
#include <terminal_strings.h>

using namespace CFL;

template <typename T>
void
print_traits(const T& t)
{
  std::cout << t.traits.rank << ' ' << t.traits.dim;
  std::cout << " Test function? " << Traits::is_test_function_set<T>::value << std::endl;
}

int
main()
{
  TerminalString<0, 3> u("u");
  auto Du = grad(u);
  auto DDu = grad(Du);

  std::cout << "Traits   u: ";
  print_traits(u);
  std::cout << "Traits  Du: ";
  print_traits(Du);
  std::cout << "Traits DDu: ";
  print_traits(DDu);

  std::cout << u.latex(std::array<int, 3>({ 0, 0, 0 })) << std::endl;
  std::cout << Du.latex(std::array<int, 3>({ 0, 0, 0 }), 0) << std::endl;
  std::cout << Du.latex(std::array<int, 3>({ 0, 0, 0 }), 1) << std::endl;
  std::cout << Du.latex(std::array<int, 3>({ 0, 0, 0 }), 2) << std::endl;

  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 0) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 0) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 2, 0) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 1) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 1) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 2, 1) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 2) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 2) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 2, 2) << std::endl;

  TerminalString<2, 3, true> v("v");
  auto Dv = grad(v);
  auto DDv = grad(Dv);

  std::cout << "Traits   v: ";
  print_traits(v);
  std::cout << "Traits  Dv: ";
  print_traits(Dv);
  std::cout << "Traits DDv: ";
  print_traits(DDv);

  std::cout << v.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 2) << std::endl;
  std::cout << Dv.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 1, 2) << std::endl;
  std::cout << Dv.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 1, 2) << std::endl;
  std::cout << Dv.latex(std::array<int, 3>({ 0, 0, 0 }), 2, 1, 2) << std::endl;

  return 0;
}
