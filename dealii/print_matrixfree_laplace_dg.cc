// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#define DEBUG_OUTPUT

#include "matrixfree_data.h"
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <cfl/cfl.h>
#include <cfl/dealii_matrixfree.h>
#include <cfl/forms.h>

// To generate a reference solution
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include <fstream>

using namespace dealii;
using namespace CFL;
using namespace CFL::dealii::MatrixFree;

namespace Latex
{
std::string
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

/**
 * Top level base class for FE Functions, should never be constructed
 * Defined for safety reasons
 *
 */
template <class Derived>
class FEFunctionBaseBase
{
public:
  // This class should never be constructed
  FEFunctionBaseBase() = delete;
};

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
  double scalar_factor = 1.;

  /**
   * Default constructor
   *
   */
  explicit FEFunctionBaseBase(double new_factor = 1.)
    : scalar_factor(new_factor)
  {
  }

  std::string value(const std::vector<std::string>& function_names) const = delete;

  /**
   * Allows to scale an FE function with a arithmetic value
   *
   */
  template <typename Number>
  typename std::enable_if_t<std::is_arithmetic<Number>::value, Derived<rank, dim, idx>> operator*(
    const Number scalar_factor_) const
  {
    return Derived<rank, dim, idx>(scalar_factor * scalar_factor_);
  }

  /**
   * Allows to negate an FE function
   *
   */
  Derived<rank, dim, idx>
  operator-() const
  {
    const Derived<rank, dim, idx> newfunction(-scalar_factor);
    return newfunction;
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

  FEFunctionInteriorFace() {}

  std::string
  value(const std::vector<std::string>& function_names) const
  {
    return double_to_string(Base::scalar_factor) + function_names[idx] + "^+";
  }

  /**
   * Wrapper around set_evaluation_flags_face function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 0),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEFunction is vector valued or "
                  "the FEFunction is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags_face<Base::index>(true, false, false);
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

  /**
   * Wrapper around set_evaluation_flags_face function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 0),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEFunction is vector valued or "
                  "the FEFunction is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags_face<Base::index>(true, false, false);
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

  /**
   * Wrapper around set_evaluation_flags function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 0),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEFunction is vector valued or "
                  "the FEFunction is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags_face<Base::index>(false, true, false);
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

  /**
   * Wrapper around set_evaluation_flags function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 0),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEFunction is vector valued or "
                  "the FEFunction is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags_face<Base::index>(false, true, false);
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

  /**
   * Wrapper around set_evaluation_flags function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 0),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEFunction is vector valued or "
                  "the FEFunction is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags<Base::index>(true, false, false);
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

  explicit FEDivergence(const FEFunction<rank + 1, dim, idx>& fefunction)
    : FEDivergence(fefunction.name(), fefunction.scalar_factor)
  {
  }

  /**
   * Wrapper around set_evaluation_flags function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert(FEEvaluation::template rank<Base::index>() > 0,
                  "The proposed FiniteElement has to be "
                  "vector valued for using FEDivergence!");
    phi.template set_evaluation_flags<Base::index>(false, true, false);
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
  static constexpr unsigned int index = FEFunctionType::idx;

  explicit FELiftDivergence(const FEFunctionType fe_function)
    : fefunction(std::move(fe_function))
  {
  }

  auto
  operator-() const
  {
    return FELiftDivergence(-fefunction);
  }

  template <typename Number>
  typename std::enable_if_t<std::is_arithmetic<Number>::value, FELiftDivergence<FEFunctionType>>
  operator*(const Number scalar_factor_) const
  {
    return FELiftDivergence<FEFunctionType>(
      FEFunctionType(fefunction.name(), fefunction.scalar_factor * scalar_factor_));
  }

  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    FEFunctionType::set_evaluation_flags(phi);
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

  /**
   * Wrapper around set_evaluation_flags function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 1),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEGradient is vector valued or "
                  "the FEGradient is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags<Base::index>(false, true, false);
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

  explicit FECurl(const FEFunction<rank - 1, dim, idx>& fefunction)
    : FECurl(fefunction.name(), fefunction.scalar_factor)
  {
  }

  /**
   * Wrapper around set_evaluation_flags function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 1),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEGradient is vector valued or "
                  "the FEGradient is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags<Base::index>(false, true, false);
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

  explicit FEGradient(const FEFunction<rank - 1, dim, idx>& fefunction)
    : FEGradient(fefunction.name(), fefunction.scalar_factor)
  {
  }

  /**
   * Wrapper around set_evaluation_flags function of FEEvaluation
   *
   */
  /*template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 1),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEGradient is vector valued or "
                  "the FEGradient is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags<Base::index>(false, true, false);
  }*/
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

  explicit FELaplacian(const FEGradient<rank + 1, dim, idx>& fe_function)
    : FELaplacian(fe_function.name(), fe_function.scalar_factor)
  {
  }

  /**
   * Wrapper around set_evaluation_flags function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 2),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEHessian is vector valued or "
                  "the FEHessian is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags<Base::index>(false, false, true);
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

  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 2),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEHessian is vector valued or "
                  "the FEHessian is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags<Base::index>(false, false, true);
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

  explicit FEHessian(const FEGradient<rank - 1, dim, idx>& fefunction)
    : FEHessian(fefunction.name(), fefunction.scalar_factor)
  {
  }

  /**
   * Wrapper around set_evaluation_flags function of FEEvaluation
   *
   */
  template <class FEEvaluation>
  static void
  set_evaluation_flags(FEEvaluation& phi)
  {
    static_assert((FEEvaluation::template rank<Base::index>() > 0) ==
                    (Base::TensorTraits::rank > 2),
                  "Either the proposed FiniteElement is scalar valued "
                  "and the FEHessian is vector valued or "
                  "the FEHessian is scalar valued and "
                  "the FiniteElement is vector valued!");
    phi.template set_evaluation_flags<Base::index>(false, false, true);
  }
};

/**
 * Utility function to return a FEGradient object
 * given a FEFunction
 *
 */
template <int rank, int dim, unsigned int idx>
FEGradient<rank + 1, dim, idx>
grad(const FEFunction<rank, dim, idx>& f)
{
  return FEGradient<rank + 1, dim, idx>(f);
}

/**
 * Utility function to return a FEDivergence object
 * given a FEFunction
 *
 */
template <int rank, int dim, unsigned int idx>
FEDivergence<rank - 1, dim, idx>
div(const FEFunction<rank, dim, idx>& f)
{
  return FEDivergence<rank - 1, dim, idx>(f);
}

/**
 * Utility function to return a FEHessian object
 * given a FEGradient
 *
 */
template <int rank, int dim, unsigned int idx>
FEHessian<rank + 1, dim, idx>
grad(const FEGradient<rank, dim, idx>& f)
{
  return FEHessian<rank + 1, dim, idx>(f);
}

/**
 * Utility function to return a FELaplacian object
 * given a FEGradient
 *
 */
template <int rank, int dim, unsigned int idx>
FELaplacian<rank - 1, dim, idx>
div(const FEGradient<rank, dim, idx>& f)
{
  return FELaplacian<rank - 1, dim, idx>(f);
}
}

template <auto... ints>
auto
transform(const FEFunctionInteriorFace<ints...>& f)
{
  return Latex::FEFunctionInteriorFace<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FEFunctionExteriorFace<ints...>& f)
{
  return Latex::FEFunctionExteriorFace<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FENormalGradientInteriorFace<ints...>& f)
{
  return Latex::FENormalGradientInteriorFace<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FENormalGradientExteriorFace<ints...>& f)
{
  return Latex::FENormalGradientExteriorFace<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FEFunction<ints...>& f)
{
  return Latex::FEFunction<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FEDivergence<ints...>& f)
{
  return Latex::FEDivergence<ints...>(f.scalar_factor);
}
template <class Type>
Latex::FELiftDivergence<Type> transform(const FELiftDivergence<Type>&);
template <auto... ints>
auto
transform(const FESymmetricGradient<ints...>& f)
{
  return Latex::FESymmetricGradient<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FECurl<ints...>& f)
{
  return Latex::FECurl<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FEGradient<ints...>& f)
{
  return Latex::FEGradient<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FEDiagonalHessian<ints...>& f)
{
  return Latex::FEDiagonalHessian<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FEHessian<ints...>& f)
{
  return Latex::FEHessian<ints...>(f.scalar_factor);
}
template <auto... ints>
auto
transform(const FELaplacian<ints...>& f)
{
  return Latex::FEFunction<ints...>(f.scalar_factor);
}

template <class... Types>
auto
transform(const SumFEFunctions<Types...>& f)
{
  return SumFEFunctions<decltype(transform(std::declval<Types>()))...>(f);
}

template <class LatexTest, class LatexExpr, FormKind kind_of_form>
class LatexForm
{
public:
  template <class Test, class Expr, typename NumberType>
  LatexForm(const Form<Test, Expr, kind_of_form, NumberType> f)
    : expr(transform(f.expr))
  {
  }

  LatexForm() {}

  std::string
  print(const std::vector<std::string>& function_names,
        const std::vector<std::string>& expression_names) const
  {
    const std::string domain = []() {
      switch (kind_of_form)
      {
        case FormKind::cell:
          return R"(_\Omega)";
          break;
        case FormKind::face:
          return R"(_F)";
          break;
        case FormKind::boundary:
          return R"(_{\partial \Omega})";
          break;
        default:
          Assert(false, ::ExcInternalError());
      }
    }();
    return "(" + expr.value(function_names) + "," + test.print(expression_names) + ")" + domain;
  }

private:
  const LatexExpr expr;
  LatexTest test;
};

template <typename... Types>
class LatexForms;

template <typename FormType, typename... FormTypes>
class LatexForms<FormType, FormTypes...> : public LatexForms<FormTypes...>
{
public:
  template <class OtherType, class... OtherTypes,
            typename std::enable_if<sizeof...(OtherTypes) == sizeof...(FormTypes)>::type* = nullptr>
  LatexForms(const Forms<OtherType, OtherTypes...>& f)
    : LatexForms<FormTypes...>(static_cast<Forms<OtherTypes...>>(f))
    , form(f.get_form())
  {
  }

  std::string
  print(const std::vector<std::string>& function_names,
        const std::vector<std::string>& expression_names) const
  {
    return form.print(function_names, expression_names) + "+" +
           LatexForms<FormTypes...>::print(function_names, expression_names);
  }

private:
  FormType form;
};

template <class Test, class Expr, FormKind kind_of_form>
class LatexForms<LatexForm<Test, Expr, kind_of_form>>
{
public:
  template <class OtherExpr, typename NumberType>
  LatexForms(const Forms<Form<Test, OtherExpr, kind_of_form, NumberType>>& f)
    : form(f.get_form())
  {
  }

  template <typename NumberType>
  void
  initialize(const Form<Test, Expr, kind_of_form, NumberType>)
  {
  }

  std::string
  print(const std::vector<std::string>& function_names,
        const std::vector<std::string>& expression_names) const
  {
    return form.print(function_names, expression_names);
  }

private:
  LatexForm<Test, Expr, kind_of_form> form;
};

template <class Test, class Expr, FormKind kind_of_form, typename NumberType>
LatexForm<Test, decltype(transform(std::declval<Expr>())), kind_of_form> transform_latex(
  const Form<Test, Expr, kind_of_form, NumberType>&);

template <typename... Types>
auto
transform_latex(const Forms<Types...>& f)
{
  auto new_form = LatexForms<decltype(transform_latex(std::declval<Types>()))...>(f);
  // static for  new_form.initialize_recursively
  return new_form;
}

template <int dim, unsigned int degree>
void
run()
{
  FE_DGQ<dim> fe_u(degree);

  FEData<FE_DGQ, degree, 1, dim, 0, 1> fedata1(fe_u);
  FEDataFace<FE_DGQ, degree, 1, dim, 0, 1> fedata_face1(fe_u);
  auto fe_datas = (fedata_face1, fedata1);

  std::vector<FiniteElement<dim>*> fes;
  fes.push_back(&fe_u);

  TestFunction<0, 1, 0> v;
  auto Dv = grad(v);
  TestFunctionInteriorFace<0, 1, 0> v_p;
  TestFunctionExteriorFace<0, 1, 0> v_m;
  TestNormalGradientInteriorFace<0, 1, 0> Dnv_p;
  TestNormalGradientExteriorFace<0, 1, 0> Dnv_m;

  FEFunction<0, 1, 0> u("u");
  auto Du = grad(u);
  FEFunctionInteriorFace<0, 1, 0> u_p("u+");
  FEFunctionExteriorFace<0, 1, 0> u_m("u-");
  FENormalGradientInteriorFace<0, 1, 0> Dnu_p("u+");
  FENormalGradientExteriorFace<0, 1, 0> Dnu_m("u-");

  auto cell = form(Du, Dv);

  auto flux = u_p - u_m;
  auto flux_grad = Dnu_p - Dnu_m;

  auto flux1 = -face_form(flux, Dnv_p) + face_form(flux, Dnv_m);
  auto flux2 = face_form(-flux + .5 * flux_grad, v_p) - face_form(-flux + .5 * flux_grad, v_m);

  auto boundary1 = boundary_form(2. * u_p - Dnu_p, v_p);
  auto boundary3 = -boundary_form(u_p, Dnv_p);

  auto face = -flux2 + .5 * flux1;
  auto f = cell + face + boundary1 + boundary3;

  std::vector<std::string> function_names(1, "u");
  std::vector<std::string> test_names(1, "v");
  std::cout << transform_latex(f).print(function_names, test_names) << std::endl;
}

int
main(int /*argc*/, char** /*argv*/)
{
  deallog.depth_console(10);
  //::dealii::MultithreadInfo::set_thread_limit( (argc > 1) ? atoi(argv[1]) : 1);
  // std::cout << ::dealii::MultithreadInfo::n_threads() << std::endl;
  try
  {
    constexpr unsigned int degree = 1;
    run<2, degree>();
  }
  catch (std::exception& exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
