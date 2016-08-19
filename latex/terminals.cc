
#include <iostream>
#include <terminal_strings.h>

int main()
{
  TerminalString<0,3> u("u");

  std::cout << u.latex(std::array<int,0>(), std::array<int,3>({0,0,0})) << std::endl;
  std::cout << u.latex(std::array<int,0>(), std::array<int,3>({0,1,0})) << std::endl;
  std::cout << u.latex(std::array<int,0>(), std::array<int,3>({1,1,0})) << std::endl;
  std::cout << u.latex(std::array<int,0>(), std::array<int,3>({1,2,3})) << std::endl;

  TerminalString<2,3> v("v");

  std::cout << v.latex(std::array<int,2>({0,1}), std::array<int,3>({0,0,0})) << std::endl;
  std::cout << v.latex(std::array<int,2>({2,2}), std::array<int,3>({0,1,0})) << std::endl;
  std::cout << v.latex(std::array<int,2>({0,0}), std::array<int,3>({1,1,0})) << std::endl;
  std::cout << v.latex(std::array<int,2>({2,1}), std::array<int,3>({1,2,3})) << std::endl;

  return 0;
}
