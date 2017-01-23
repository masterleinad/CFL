// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include <cfl/terminal_strings.h>
#include <iostream>
#include <stdexcept>

using namespace CFL;

template <typename T>
void
print_test(const T& t, bool check)
{
  std::cout << "Test function? " << Traits::is_test_function_set<T>::value << std::endl;
  if (Traits::is_test_function_set<T>::value != check)
    throw std::logic_error(std::string("Wrong value for test function set: ") +
                           std::to_string(Traits::is_test_function_set<T>::value) + " should be " +
                           std::to_string(check));
}

void
check_string(const std::string& str, const std::string& check)
{
  std::cout << str << std::endl;
  if (str != check)
    throw std::logic_error(std::string("Wrong string: \"") + str + "\" should be \"" + check +
                           "\"");
}

int
main()
{
  TerminalString<0, 3> u("u");
  print_test(u, false);
  check_string(u.latex(), "u");
  check_string(u.latex(std::array<int, 3>({ 0, 0, 0 })), "u");
  check_string(u.latex(std::array<int, 3>({ 0, 1, 0 })), "\\partial_{1}u");
  check_string(u.latex(std::array<int, 3>({ 1, 1, 0 })), "\\partial_{0}\\partial_{1}u");
  check_string(u.latex(std::array<int, 3>({ 1, 2, 3 })),
               "\\partial_{0}\\partial_{1}^{2}\\partial_{2}^{3}u");

  TerminalString<1, 2> w("w");
  check_string(w.latex(0), "w_{0}");
  check_string(w.latex(1), "w_{1}");

  TerminalString<2, 3, true> v("v");
  print_test(v, true);
  check_string(v.latex(std::array<int, 3>({ 0, 0, 0 }), 0, 1), "v_{10}");
  check_string(v.latex(0, 1), "v_{10}");

  check_string(v.latex(std::array<int, 3>({ 0, 1, 0 }), 2, 2), "\\partial_{1}v_{22}");
  check_string(v.latex(std::array<int, 3>({ 1, 1, 0 }), 0, 0), "\\partial_{0}\\partial_{1}v_{00}");
  check_string(v.latex(std::array<int, 3>({ 1, 2, 3 }), 2, 1),
               "\\partial_{0}\\partial_{1}^{2}\\partial_{2}^{3}v_{12}");

  return 0;
}
