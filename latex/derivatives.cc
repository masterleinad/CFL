
#include <iostream>
#include <terminal_strings.h>
#include <derivatives.h>

using namespace CFL;

int
main()
{
  TerminalString<0, 3> u("u");
  auto Du = grad(u);
  auto DDu = grad(Du);

  std::cout << "Traits   u: " << u.traits.rank << ' ' << u.traits.dim << std::endl;
  std::cout << "Traits  Du: " << Du.traits.rank << ' ' << Du.traits.dim << std::endl;
  std::cout << "Traits DDu: " << DDu.traits.rank << ' ' << DDu.traits.dim << std::endl;

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

  TerminalString<2, 3> v("v");
  auto Dv = grad(v);

  std::cout << "Traits   v: " << v.traits.rank << ' ' << v.traits.dim << std::endl;
  std::cout << "Traits  Dv: " << Dv.traits.rank << ' ' << Dv.traits.dim << std::endl;
  std::cout << "Traits DDv: " << DDv.traits.rank << ' ' << DDv.traits.dim << std::endl;

  std::cout << v.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 2) << std::endl;
  std::cout << Dv.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 1, 2) << std::endl;
  std::cout << Dv.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 1, 2) << std::endl;
  std::cout << Dv.latex(std::array<int, 3>({ 0, 0, 0 }), 2, 1, 2) << std::endl;

  return 0;
}
