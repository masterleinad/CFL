// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include <cfl/traits.h>
#include <iostream>
#include <latex/fefunctions.h>
#include <stdexcept>

using namespace CFL;

template <typename T>
void
print_test(const T&, const ObjectType& check)
{
  auto to_string = [](const ObjectType& obj) {
    switch (obj)
    {
      case ObjectType::none:
        return "none";
        break;
      case ObjectType::cell:
        return "cell";
        break;
      case ObjectType::face:
        return "face";
        break;
      default:
        assert(false);
        return "";
    }
  };
  const std::string set_type_string = to_string(Traits::test_function_set_type<T>::value);
  const std::string check_string = to_string(check);
  std::cout << "Test function? " << set_type_string << std::endl;
  if (set_type_string != check_string)
    throw std::logic_error(std::string("Wrong value for test function set: ") + set_type_string +
                           " should be " + check_string);
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
  Base::FEFunction<0, 3, 0> u("u");
  Base::FEFunctionInteriorFace<1, 2, 1> p("p");
  std::vector<std::string> function_names{ "u", "p" };
  Base::TestFunction<2, 3, 0> v;
  Base::TestFunctionInteriorFace<2, 3, 1> q;
  std::vector<std::string> test_names{ "v", "q" };

  print_test(u, ObjectType::none);
  check_string(Latex::transform(u).value(function_names), "u");
  print_test(p, ObjectType::none);
  check_string(Latex::transform(p).value(function_names), "p^+");

  print_test(v, ObjectType::cell);
  check_string(Latex::transform(v).submit(test_names), "v");
  print_test(q, ObjectType::face);
  check_string(Latex::transform(q).submit(test_names), "q^+");

  return 0;
}
