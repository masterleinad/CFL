// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include <cfl/latex/fefunctions.h>

#include <iostream>
#include <stdexcept>

using namespace CFL;

template <int expected, class T>
constexpr void
check_rank(const T&)
{
  static_assert(T::TensorTraits::rank == expected, "Rank not as expected");
}

int
main()
{
  {
    Base::FEFunction<0, 3, 0> u;
    auto Du = grad(u);
    auto DDu = grad(Du);

    std::vector<std::string> function_names{ "u" };

    check_rank<0>(u);
    check_rank<1>(Du);
    check_rank<2>(DDu);

    std::cout << Latex::transform(u).value(function_names) << std::endl;
    std::cout << Latex::transform(Du).value(function_names) << std::endl;
    std::cout << Latex::transform(DDu).value(function_names) << std::endl;
  }
  {
    Base::TestFunction<2, 3, 0> v;
    auto Dv = grad(v);
    auto DDv = grad(Dv);

    std::vector<std::string> test_names{ "v" };

    check_rank<2>(v);
    check_rank<3>(Dv);
    check_rank<4>(DDv);

    static_assert(std::is_base_of<Base::TestFunctionBaseBase<decltype(v)>, decltype(v)>::value);

    std::cout << Latex::transform(v).submit(test_names) << std::endl;
    std::cout << Latex::transform(Dv).submit(test_names) << std::endl;
    std::cout << Latex::transform(DDv).submit(test_names) << std::endl;
  }

  return 0;
}
