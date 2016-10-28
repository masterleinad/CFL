
#include <iostream>
#include <terminal_strings.h>

using namespace CFL;

template <typename T>
void
print_test(const T& t)
{
  std::cout << "Test function? " << Traits::is_test_function_set<T>::value << std::endl;
}

int
main()
{
  TerminalString<0, 3> u("u");
  print_test(u);
  std::cout << u.latex(std::array<int, 3>({ 0, 0, 0 })) << std::endl;
  std::cout << u.latex(std::array<int, 3>({ 0, 1, 0 })) << std::endl;
  std::cout << u.latex(std::array<int, 3>({ 1, 1, 0 })) << std::endl;
  std::cout << u.latex(std::array<int, 3>({ 1, 2, 3 })) << std::endl;

  TerminalString<2, 3, true> v("v");
  print_test(v);
  std::cout << v.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 1) << std::endl;
  std::cout << v.latex(std::array<int, 3>({ 0, 1, 0 }), 2, 2) << std::endl;
  std::cout << v.latex(std::array<int, 3>({ 1, 1, 0 }), 0, 0) << std::endl;
  std::cout << v.latex(std::array<int, 3>({ 1, 2, 3 }), 2, 1) << std::endl;

  return 0;
}
