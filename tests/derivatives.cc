// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include <cfl/derivatives.h>
#include <cfl/terminal_strings.h>
#include <iostream>
#include <stdexcept>

using namespace CFL;

template <int expected, class T>
constexpr void
check_rank(const T& t)
{
  static_assert(T::TensorTraits::rank == expected, "Rank not as expected");
}

int
main()
{
  TerminalString<0, 3> u("u");
  auto Du = grad(u);
  auto DDu = grad(Du);

  check_rank<0>(u);
  check_rank<1>(Du);
  check_rank<2>(DDu);

  std::cout << u.latex() << std::endl;
  std::cout << Du.latex(0) << " = " << Du.latex(0) << std::endl;
  std::cout << Du.latex(1) << " = " << Du.latex(1) << std::endl;
  std::cout << Du.latex(2) << " = " << Du.latex(2) << std::endl;

  std::cout << DDu.latex(0, 0) << " = " << DDu.latex(0, 0) << std::endl;
  std::cout << DDu.latex(1, 0) << " = " << DDu.latex(1, 0) << std::endl;
  std::cout << DDu.latex(2, 0) << " = " << DDu.latex(2, 0) << std::endl;
  std::cout << DDu.latex(0, 1) << std::endl;
  std::cout << DDu.latex(1, 1) << std::endl;
  std::cout << DDu.latex(2, 1) << std::endl;
  std::cout << DDu.latex(0, 2) << std::endl;
  std::cout << DDu.latex(1, 2) << std::endl;
  std::cout << DDu.latex(2, 2) << std::endl;

  TerminalString<2, 3, true> v("v");
  auto Dv = grad(v);
  auto DDv = grad(Dv);

  check_rank<2>(v);
  check_rank<3>(Dv);
  check_rank<4>(DDv);

  std::cout << v.latex(1, 2) << " = " << v.latex(1, 2) << std::endl;
  std::cout << Dv.latex(0, 1, 2) << " = " << Dv.latex(0, 1, 2) << std::endl;
  std::cout << Dv.latex(1, 1, 2) << " = " << Dv.latex(1, 1, 2) << std::endl;
  std::cout << Dv.latex(2, 1, 2) << " = " << Dv.latex(2, 1, 2) << std::endl;

  return 0;
}
