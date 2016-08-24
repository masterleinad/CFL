
#include <iostream>
#include <terminal_strings.h>

using namespace CFL;

int
main()
{
  TerminalString<0, 3> u("u");

  std::cout << u.latex(std::array<int, 3>({ 0, 0, 0 })) << std::endl;
  std::cout << u.latex(std::array<int, 3>({ 0, 1, 0 })) << std::endl;
  std::cout << u.latex(std::array<int, 3>({ 1, 1, 0 })) << std::endl;
  std::cout << u.latex(std::array<int, 3>({ 1, 2, 3 })) << std::endl;

  TerminalString<2, 3> v("v");

  std::cout << v.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 1) << std::endl;
  std::cout << v.latex(std::array<int, 3>({ 0, 1, 0 }), 2, 2) << std::endl;
  std::cout << v.latex(std::array<int, 3>({ 1, 1, 0 }), 0, 0) << std::endl;
  std::cout << v.latex(std::array<int, 3>({ 1, 2, 3 }), 2, 1) << std::endl;

  return 0;
}
