#ifndef LATEX_FEFUNCTIONS_H
#define LATEX_FEFUNCTIONS_H

#include <cfl/base/fefunctions.h>

#include <string>
#include <vector>

namespace CFL::Latex
{
namespace
{
  inline std::string
  double_to_string(double d)
  {
    std::string return_string = std::to_string(d);
    return_string.erase(return_string.find_last_not_of('0') + 1, std::string::npos);
    return_string.erase(return_string.find_last_not_of('.') + 1, std::string::npos);
    if (return_string == "1")
      return "";
    if (return_string == "-1")
      return "-";
    return return_string;
  }
}

/**
 * Top level base class for FE Functions, should never be constructed
 * Defined for safety reasons
 *
 */
template <class Derived>
class FEFunctionBaseBase;

/**
 * Top level base class for FE Functions
 * See \ref FEFunctionBase for more details
 *
 */
template <template <int, int, unsigned int> class Derived, int rank, int dim, unsigned int idx>
class FEFunctionBaseBase<Derived<rank, dim, idx>>
{
public:
  using TensorTraits = Traits::Tensor<rank, dim>;
  static constexpr unsigned int index = idx;
  const double scalar_factor = 1.;

  /**
   * Default constructor
   *
   */
  constexpr explicit FEFunctionBaseBase(double new_factor = 1.)
    : scalar_factor(new_factor)
  {
  }
};

/**
 * Base class for all FE Function classes
 * @note
 * <li> See that this is a templatized class, with template parameter
 * as the actual derived class. This might look like CRTP pattern, but
 * its not since the base class is not trying to use static polymorphism.
 * This way of implementation allows us to clearly structure our class
 * heirarchy and collect the values of <code> index </code>, and Tensor
 * traits in a single place
 * <li> Also note that because this is a template class, the actual base
 * class which is created after template specialization will be different
 * for each FE Function class. This is different from traditional non-
 * template base-derived heirarchy where all derived classes have common
 * base class.
 */
template <class Derived>
class FEFunctionBase : public FEFunctionBaseBase<Derived>
{
  using FEFunctionBaseBase<Derived>::FEFunctionBaseBase;
};

/**
 * Top level base class for FE Functions on Face
 * See \ref TestFunctionBase for more details
 *
 */
template <class Derived>
class FEFunctionFaceBase : public FEFunctionBaseBase<Derived>
{
  using FEFunctionBaseBase<Derived>::FEFunctionBaseBase;
};

/**
 * FE Function which provides evaluation on interior faces
 * in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FEFunctionInteriorFace final
  : public FEFunctionFaceBase<FEFunctionInteriorFace<rank, dim, idx>>
{
public:
  using Base = FEFunctionFaceBase<FEFunctionInteriorFace<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + function_names[idx] + "^+";
  }
};

/**
 * FE Function which provides evaluation on exterior faces
 * in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FEFunctionExteriorFace final
  : public FEFunctionFaceBase<FEFunctionExteriorFace<rank, dim, idx>>
{
public:
  using Base = FEFunctionFaceBase<FEFunctionExteriorFace<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + function_names[idx] + "^-";
  }
};

/**
 * FE Function which provides gradient evaluation on interior faces
 * in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FENormalGradientInteriorFace final
  : public FEFunctionFaceBase<FENormalGradientInteriorFace<rank, dim, idx>>
{
public:
  using Base = FEFunctionFaceBase<FENormalGradientInteriorFace<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + R"(\boldsymbol{n}^+\cdot\nabla )" +
           function_names[idx] + "^+";
  }
};

/**
 * FE Function which provides gradient evaluation on exterior faces
 * in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FENormalGradientExteriorFace final
  : public FEFunctionFaceBase<FENormalGradientExteriorFace<rank, dim, idx>>
{
public:
  using Base = FEFunctionFaceBase<FENormalGradientExteriorFace<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + R"(\boldsymbol{n}^-\cdot\nabla )" +
           function_names[idx] + "^-";
  }
};

/**
 * FE Function which provides evaluation on cell in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FEFunction final : public FEFunctionBase<FEFunction<rank, dim, idx>>
{
public:
  using Base = FEFunctionBase<FEFunction<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + function_names[idx];
  }
};

/**
 * FE Function which provides divergence evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FEDivergence final : public FEFunctionBase<FEDivergence<rank, dim, idx>>
{
public:
  using Base = FEFunctionBase<FEDivergence<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + R"(\nabla\cdot )" + function_names[idx] + "^+";
  }
};

/**
 * TBD
 *
 */
template <class FEFunctionType>
class FELiftDivergence final
{
private:
  const FEFunctionType fefunction;

public:
  using TensorTraits =
    Traits::Tensor<FEFunctionType::TensorTraits::rank + 2, FEFunctionType::TensorTraits::dim>;

  explicit FELiftDivergence(FEFunctionType fe_function)
    : fefunction(std::move(fe_function))
  {
  }

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return fefunction.value(function_names);
  }
};

/**
 * FE Function which provides Symmetric Gradient evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FESymmetricGradient final : public FEFunctionBase<FESymmetricGradient<rank, dim, idx>>
{
public:
  using Base = FEFunctionBase<FESymmetricGradient<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor / 2.) + R"(((\nabla )" + function_names[idx] + "+" +
           R"((\nabla )" + function_names[idx] + ")^T))";
  }
};

/**
 * FE Function which provides curl evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FECurl final : public FEFunctionBase<FECurl<rank, dim, idx>>
{
public:
  using Base = FEFunctionBase<FECurl<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + R"(\nabla\times )" + function_names[idx];
  }
};

/**
 * FE Function which provides gradient evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FEGradient final : public FEFunctionBase<FEGradient<rank, dim, idx>>
{
public:
  using Base = FEFunctionBase<FEGradient<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + R"(\nabla )" + function_names[idx];
  }
};

/**
 * FE Function which provides Laplacian evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FELaplacian final : public FEFunctionBase<FELaplacian<rank, dim, idx>>
{
public:
  using Base = FEFunctionBase<FELaplacian<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + R"(\Delta )" + function_names[idx];
  }
};

/**
 * TBD
 *
 */
template <int rank, int dim, unsigned int idx>
class FEDiagonalHessian final : public FEFunctionBase<FEDiagonalHessian<rank, dim, idx>>
{
public:
  using Base = FEFunctionBase<FEDiagonalHessian<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + R"(I.\nabla \nabla )" + function_names[idx];
  }
};

/**
 * FE Function which provides Hessian evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class FEHessian final : public FEFunctionBase<FEHessian<rank, dim, idx>>
{
public:
  using Base = FEFunctionBase<FEHessian<rank, dim, idx>>;
  // inherit constructors
  using Base::Base;

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + R"(\nabla \nabla )" + function_names[idx];
  }
};

template <auto... ints>
constexpr auto
transform(const Base::FEFunctionInteriorFace<ints...>& f)
{
  return FEFunctionInteriorFace<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FEFunctionExteriorFace<ints...>& f)
{
  return FEFunctionExteriorFace<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FENormalGradientInteriorFace<ints...>& f)
{
  return FENormalGradientInteriorFace<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FENormalGradientExteriorFace<ints...>& f)
{
  return FENormalGradientExteriorFace<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FEFunction<ints...>& f)
{
  return FEFunction<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FEDivergence<ints...>& f)
{
  return FEDivergence<ints...>(f.scalar_factor);
}
template <class Type>
constexpr auto
transform(const Base::FELiftDivergence<Type>& f)
{
  return FELiftDivergence<decltype(transform(std::declval<Type>()))>(transform(f.get_fefunction()));
}
template <auto... ints>
constexpr auto
transform(const Base::FESymmetricGradient<ints...>& f)
{
  return FESymmetricGradient<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FECurl<ints...>& f)
{
  return FECurl<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FEGradient<ints...>& f)
{
  return FEGradient<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FEDiagonalHessian<ints...>& f)
{
  return FEDiagonalHessian<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FEHessian<ints...>& f)
{
  return FEHessian<ints...>(f.scalar_factor);
}
template <auto... ints>
constexpr auto
transform(const Base::FELaplacian<ints...>& f)
{
  return FELaplacian<ints...>(f.scalar_factor);
}

template <class... Types>
constexpr auto
transform(const Base::SumFEFunctions<Types...>& f)
{
  return Base::SumFEFunctions<decltype(transform(std::declval<Types>()))...>(f);
}

template <class... Types>
constexpr auto
transform(const Base::ProductFEFunctions<Types...>& f)
{
  return Base::ProductFEFunctions<decltype(transform(std::declval<Types>()))...>(f);
}

/**
 * Top level base class for Test Functions, should never be constructed
 * Defined for safety reasons
 *
 */
template <class T>
class TestFunctionBaseBase
{
public:
  // This class should never be constructed
  TestFunctionBaseBase() = delete;
};

/**
 * Top level base class for Test Functions
 * See \ref TestFunctionBase for more details
 *
 */
template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
class TestFunctionBaseBase<T<rank, dim, idx>>
{
public:
  using TensorTraits = Traits::Tensor<rank, dim>;

  static constexpr unsigned int index = idx;
  static constexpr bool scalar_valued = (TensorTraits::rank == 0);
};

/**
 * Base class for all Test Function classes
 * @note
 * <li> See that this is a templatized class, with template parameter
 * as the actual derived class. This might look like CRTP pattern, but
 * its not since the base class is not trying to use static polymorphism.
 * This way of implementation allows us to clearly structure our class
 * heirarchy and collect the values of <code> index </code>, and Tensor
 * traits in a single place
 * <li> Also note that because this is a template class, the actual base
 * class which is created after template specialization will be different
 * for each Test Function class. This is different from traditional non-
 * template base-derived heirarchy where all derived classes have common
 * base class.
 */
template <class Derived>
class TestFunctionBase : public TestFunctionBaseBase<Derived>
{
  using TestFunctionBaseBase<Derived>::TestFunctionBaseBase;
};

/**
 * Top level base class for Test Functions on Face
 * See \ref TestFunctionBase for more details
 *
 */
template <class Derived>
class TestFunctionFaceBase : public TestFunctionBaseBase<Derived>
{
  using TestFunctionBaseBase<Derived>::TestFunctionBaseBase;
};

/**
 * Test Function which provides evaluation on interior faces
 * in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestFunctionInteriorFace final
  : public TestFunctionFaceBase<TestFunctionInteriorFace<rank, dim, idx>>
{
public:
  using Base = TestFunctionFaceBase<TestFunctionInteriorFace<rank, dim, idx>>;

  static std::string
  submit(const std::vector<std::string>& function_names)
  {
    return function_names[idx] + "^+";
  }
};

/**
 * Test Function which provides evaluation on exterior faces (boundary)
 * in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestFunctionExteriorFace final
  : public TestFunctionFaceBase<TestFunctionExteriorFace<rank, dim, idx>>
{
public:
  using Base = TestFunctionFaceBase<TestFunctionExteriorFace<rank, dim, idx>>;

  static std::string
  submit(const std::vector<std::string>& function_names)
  {
    return function_names[idx] + "^-";
  }
};

/**
 * Test Function which provides evaluation of gradients on interior faces
 * in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestNormalGradientInteriorFace final
  : public TestFunctionFaceBase<TestNormalGradientInteriorFace<rank, dim, idx>>
{
public:
  using Base = TestFunctionFaceBase<TestNormalGradientInteriorFace<rank, dim, idx>>;

  static std::string
  submit(const std::vector<std::string>& function_names)
  {
    return R"(\boldsymbol{n}^+\cdot\nabla )" + function_names[idx] + "^+";
  }
};

/**
 * Test Function which provides evaluation of gradients on exterior faces
 * in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestNormalGradientExteriorFace final
  : public TestFunctionFaceBase<TestNormalGradientExteriorFace<rank, dim, idx>>
{
public:
  using Base = TestFunctionFaceBase<TestNormalGradientExteriorFace<rank, dim, idx>>;

  static std::string
  submit(const std::vector<std::string>& function_names)
  {
    return R"(\boldsymbol{n}^-\cdot\nabla )" + function_names[idx] + "^-";
  }
};

/**
 * Test Function which provides evaluation on cell in Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestFunction final : public TestFunctionBase<TestFunction<rank, dim, idx>>
{
public:
  using Base = TestFunctionBase<TestFunction<rank, dim, idx>>;

  static std::string
  submit(const std::vector<std::string>& function_names)
  {
    return function_names[idx];
  }
};

/**
 * Test Function which provides divergence evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestDivergence final : public TestFunctionBase<TestDivergence<rank, dim, idx>>
{
public:
  using Base = TestFunctionBase<TestDivergence<rank, dim, idx>>;

  static std::string
  submit(const std::vector<std::string>& function_names)
  {
    return R"(\nabla \cdot )" + function_names[idx] + "^-";
  }
};

/**
 * Test Function which provides Symmetric Gradient evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestSymmetricGradient final : public TestFunctionBase<TestSymmetricGradient<rank, dim, idx>>
{
public:
  using Base = TestFunctionBase<TestSymmetricGradient<rank, dim, idx>>;

  std::string
  submit(const std::vector<std::string>& function_names) const
  {
    return R"(0.5*((\nabla )" + function_names[idx] + "+" +
           R"((\nabla )" + function_names[idx] + ")^T))";
  }
};

/**
 * Test Function which provides curl evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestCurl final : public TestFunctionBase<TestCurl<rank, dim, idx>>
{
public:
  using Base = TestFunctionBase<TestCurl<rank, dim, idx>>;

  std::string
  submit(const std::vector<std::string>& function_names) const
  {
    return R"(\nabla\times )" + function_names[idx];
  }
};

/**
 * Test Function which provides gradient evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestGradient final : public TestFunctionBase<TestGradient<rank, dim, idx>>
{
public:
  using Base = TestFunctionBase<TestGradient<rank, dim, idx>>;

  static std::string
  submit(const std::vector<std::string>& function_names)
  {
    return R"(\nabla )" + function_names[idx];
  }
};

/**
 * Test Function which provides hessian evaluation on cell in
 * Matrix Free context
 *
 */
template <int rank, int dim, unsigned int idx>
class TestHessian final : public TestFunctionBase<TestHessian<rank, dim, idx>>
{
public:
  using Base = TestFunctionBase<TestHessian<rank, dim, idx>>;

  std::string
  submit(const std::vector<std::string>& function_names) const
  {
    return R"(\nabla \nabla )" + function_names[idx];
  }
};

template <auto... ints>
constexpr auto
transform(const Base::TestFunctionInteriorFace<ints...>&)
{
  return TestFunctionInteriorFace<ints...>();
}
template <auto... ints>
constexpr auto
transform(const Base::TestFunctionExteriorFace<ints...>&)
{
  return TestFunctionExteriorFace<ints...>();
}
template <auto... ints>
constexpr auto
transform(const Base::TestNormalGradientInteriorFace<ints...>&)
{
  return TestNormalGradientInteriorFace<ints...>();
}
template <auto... ints>
constexpr auto
transform(const Base::TestNormalGradientExteriorFace<ints...>&)
{
  return TestNormalGradientExteriorFace<ints...>();
}
template <auto... ints>
constexpr auto
transform(const Base::TestFunction<ints...>&)
{
  return TestFunction<ints...>();
}
template <auto... ints>
constexpr auto
transform(const Base::TestDivergence<ints...>&)
{
  return TestDivergence<ints...>();
}
template <auto... ints>
constexpr auto
transform(const Base::TestSymmetricGradient<ints...>&)
{
  return TestSymmetricGradient<ints...>();
}
template <auto... ints>
constexpr auto
transform(const Base::TestCurl<ints...>&)
{
  return TestCurl<ints...>();
}
template <auto... ints>
constexpr auto
transform(const Base::TestGradient<ints...>&)
{
  return TestGradient<ints...>();
}
template <auto... ints>
constexpr auto
transform(const Base::TestHessian<ints...>&)
{
  return TestHessian<ints...>();
}

// don't transform things that have not been specified
/*template <class Type>
auto
transform(Type &&f)
{
  return std::forward<Type>(f);
}*/
}

#endif // LATEX_FEFUNCTIONS_H
