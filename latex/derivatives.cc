
#include <iostream>
#include <stdexcept>
#include <cfl/terminal_strings.h>
#include <cfl/derivatives.h>

using namespace CFL;

int
main()
{
  TerminalString<0, 3> u("u");
  auto Du = grad(u);
  auto DDu = grad(Du);

  if (u.traits.rank != 0)
    throw std::logic_error("Rank not 0!");
  if (Du.traits.rank != 1)
    throw std::logic_error("Rank not 1!");
  if (DDu.traits.rank != 2)
    throw std::logic_error("Rank not 2!");

  std::cout << u.latex(std::array<int, 3>({ 0, 0, 0 })) << std::endl;
  std::cout << Du.latex(std::array<int, 3>({ 0, 0, 0 }), 0) << " = " << Du.latex(0) << std::endl;
  std::cout << Du.latex(std::array<int, 3>({ 0, 0, 0 }), 1) << " = " << Du.latex(1) << std::endl;
  std::cout << Du.latex(std::array<int, 3>({ 0, 0, 0 }), 2) << " = " << Du.latex(2) << std::endl;

  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 0) << " = " << DDu.latex(0, 0)
            << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 0) << " = " << DDu.latex(1, 0)
            << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 2, 0) << " = " << DDu.latex(2, 0)
            << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 1) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 1) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 2, 1) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 2) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 2) << std::endl;
  std::cout << DDu.latex(std::array<int, 3>({ 0, 0, 0 }), 2, 2) << std::endl;

  TerminalString<2, 3, true> v("v");
  auto Dv = grad(v);
  auto DDv = grad(Dv);

  if (v.traits.rank != 2)
    throw std::logic_error("Rank not 2!");
  if (Dv.traits.rank != 3)
    throw std::logic_error("Rank not 3!");
  if (DDv.traits.rank != 4)
    throw std::logic_error("Rank not 4!");

  std::cout << v.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 2) << " = " << v.latex(1, 2)
            << std::endl;
  std::cout << Dv.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 1, 2) << " = " << Dv.latex(0, 1, 2)
            << std::endl;
  std::cout << Dv.latex(std::array<int, 3>({ 0, 0, 0 }), 1, 1, 2) << " = " << Dv.latex(1, 1, 2)
            << std::endl;
  std::cout << Dv.latex(std::array<int, 3>({ 0, 0, 0 }), 2, 1, 2) << " = " << Dv.latex(2, 1, 2)
            << std::endl;

  return 0;
}
