#ifndef cfl_forms_h
#define cfl_forms_h

#include <array>
#include <iostream>
#include <string>
#include <utility>

#include <tuple>

#include <deal.II/base/exceptions.h>

#include <cfl/traits.h>

namespace CFL
{
namespace
{
  enum class FormKind
  {
    cell,
    face,
    boundary
  };
}

template <FormKind, ObjectType>
constexpr bool
formkind_matches_objecttype()
{
  return false;
}

template <>
constexpr bool
formkind_matches_objecttype<FormKind::cell, ObjectType::cell>()
{
  return true;
}

template <>
constexpr bool
formkind_matches_objecttype<FormKind::face, ObjectType::face>()
{
  return true;
}

template <>
constexpr bool
formkind_matches_objecttype<FormKind::boundary, ObjectType::face>()
{
  return true;
}

template <typename... Types>
class Forms;

/**
 * A Form is an expression tested by a test function set.
 */
template <class Test, class Expr, FormKind kind_of_form, typename NumberType = double>
class Form final
{
public:
  using TestType = Test;
  const Test test;
  const Expr expr;

  static constexpr FormKind form_kind = kind_of_form;

  static constexpr unsigned int fe_number = Test::index;
  static constexpr bool integrate_value = Test::integration_flags.value;
  static constexpr bool integrate_value_exterior =
    (kind_of_form == FormKind::face) ? Test::integration_flags.value_exterior : false;
  static constexpr bool integrate_gradient = Test::integration_flags.gradient;
  static constexpr bool integrate_gradient_exterior =
    (kind_of_form == FormKind::face) ? Test::integration_flags.gradient_exterior : false;

  Form(Test test_, Expr expr_)
    : test(std::move(test_))
    , expr(std::move(expr_))
  {
    static_assert(Traits::test_function_set_type<Test>::value != ObjectType::none,
                  "The first argument must be a test function!");
    static_assert(Traits::fe_function_set_type<Expr>::value != ObjectType::none,
                  "The second argument must be a finite element function!");
    static_assert(
      formkind_matches_objecttype<kind_of_form, Traits::test_function_set_type<Test>::value>(),
      "The type of the test function must be compatible with the type of the form!");
    static_assert(
      formkind_matches_objecttype<kind_of_form, Traits::fe_function_set_type<Expr>::value>(),
      "The type of the expression must be compatible with the type of the form!");
    static_assert(Test::TensorTraits::rank == Expr::TensorTraits::rank,
                  "Test function and expression must have the same tensor rank!");
    static_assert(Test::TensorTraits::dim == Expr::TensorTraits::dim,
                  "Test function and expression must have the same dimension!");

    if constexpr(form_kind == FormKind::boundary) static_assert(
        Test::integration_flags.value_exterior == false &&
          Test::integration_flags.gradient_exterior == false,
        "A boundary form cannot have a test function associated with the neighbor of a cell!");
  }

  static constexpr void
  get_form_kinds(std::array<bool, 3>& use_objects)
  {
    switch (form_kind)
    {
      case FormKind::cell:
        use_objects[0] = true;
        break;

      case FormKind::face:
        use_objects[1] = true;
        break;

      case FormKind::boundary:
        use_objects[2] = true;
        break;

      default:
        static_assert("Invalid FormKind!");
    }
  }

  template <class FEEvaluation>
  static void
  integrate(FEEvaluation& phi)
  {
    // only to be used if there is only one form!
    phi.template integrate<fe_number>(integrate_value, integrate_gradient);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags(FEEvaluation& phi)
  {
    // only to be used if there is only one form!
    if constexpr(form_kind == FormKind::cell)
        phi.template set_integration_flags<fe_number>(integrate_value, integrate_gradient);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags_face(FEEvaluation& phi)
  {
    // only to be used if there is only one form!
    if constexpr(form_kind == FormKind::face)
        phi.template set_integration_flags_face_and_boundary<fe_number>(
          integrate_value,
          integrate_value_exterior,
          integrate_gradient,
          integrate_gradient_exterior);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags_boundary(FEEvaluation& phi)
  {
    // only to be used if there is only one form!
    if constexpr(form_kind == FormKind::boundary)
        phi.template set_integration_flags_face_and_boundary<fe_number>(
          integrate_value,
          integrate_value_exterior,
          integrate_gradient,
          integrate_gradient_exterior);
  }

  template <class FEEvaluation>
  void
  set_evaluation_flags(FEEvaluation& phi) const
  {
    // only to be used if there is only one form!
    if constexpr(form_kind == FormKind::cell) expr.set_evaluation_flags(phi);
  }

  template <class FEEvaluation>
  void
  set_evaluation_flags_face(FEEvaluation& phi) const
  {
    // only to be used if there is only one form!
    if constexpr(form_kind == FormKind::face || form_kind == FormKind::boundary)
        expr.set_evaluation_flags(phi);
  }

  template <class FEEvaluation>
  void evaluate([[maybe_unused]] FEEvaluation& phi, [[maybe_unused]] unsigned int q) const
  {
    if constexpr(form_kind == FormKind::cell)
      {
        // only to be used if there is only one form!
        const auto value = expr.value(phi, q);
        Test::submit(phi, q, value);
      }
  }

  template <class FEEvaluation>
  void evaluate_face([[maybe_unused]] FEEvaluation& phi, [[maybe_unused]] unsigned int q) const
  {
    if constexpr(form_kind == FormKind::face)
      {
        // only to be used if there is only one form!
        const auto value = expr.value(phi, q);
        Test::submit(phi, q, value);
      }
  }

  template <class FEEvaluation>
  void evaluate_boundary([[maybe_unused]] FEEvaluation& phi, [[maybe_unused]] unsigned int q) const
  {
    if constexpr(form_kind == FormKind::boundary)
      {
        // only to be used if there is only one form!
        const auto value = expr.value(phi, q);
        Test::submit(phi, q, value);
      }
  }

  template <class FEEvaluation>
  auto
  value(FEEvaluation& phi, unsigned int q) const
  {
    return expr.value(phi, q);
  }

  template <class FEEvaluation, typename ValueType>
  static void
  submit(FEEvaluation& phi, unsigned int q, const ValueType& value)
  {
    Test::submit(phi, q, value);
  }

  template <class TestNew, class ExprNew, FormKind kind_new>
  Forms<Form<Test, Expr, kind_of_form>, Form<TestNew, ExprNew, kind_new>>
  operator+(const Form<TestNew, ExprNew, kind_new>& new_form) const
  {
    return Forms<Form<Test, Expr, kind_of_form>, Form<TestNew, ExprNew, kind_new>>(*this, new_form);
  }

  template <class TestNew, class ExprNew, FormKind kind_new>
  Forms<Form<Test, Expr, kind_of_form>, Form<TestNew, ExprNew, kind_new>>
  operator-(const Form<TestNew, ExprNew, kind_new>& new_form) const
  {
    return Forms<Form<Test, Expr, kind_of_form>, Form<TestNew, ExprNew, kind_new>>(*this,
                                                                                   -new_form);
  }

  template <class... Types>
  auto
  operator+(const Forms<Types...>& old_form) const
  {
    return old_form + *this;
  }

  auto
  operator-() const
  {
    const typename std::remove_reference<decltype(*this)>::type newform(test, -expr);
    return newform;
  }

  auto operator*(const double scalar) const
  {
    const typename std::remove_reference<decltype(*this)>::type newform(test, expr * scalar);
    return newform;
  }
};

namespace Traits
{
  template <class Test, class Expr, FormKind kind_of_form>
  struct is_form<Form<Test, Expr, kind_of_form>>
  {
    const static bool value = true;
  };

  template <class Test, class Expr, FormKind kind_of_form>
  struct is_cfl_object<Form<Test, Expr, kind_of_form>>
  {
    const static bool value = true;
  };

  template <class Test1, class Expr1, FormKind kind1, class Test2, class Expr2, FormKind kind2>
  struct is_summable<Form<Test1, Expr1, kind1>, Form<Test2, Expr2, kind2>>
  {
    const static bool value = true;
  };

  template <class Test, class Expr, FormKind kind_of_form, typename... Types>
  struct is_summable<Form<Test, Expr, kind_of_form>, Forms<Types...>>
  {
    const static bool value = true;
  };

  template <class Test, class Expr, FormKind kind_of_form, typename... Types>
  struct is_summable<Forms<Types...>, Form<Test, Expr, kind_of_form>>
  {
    const static bool value = true;
  };
} // namespace Traits

template <class Test, class Expr>
typename std::enable_if<Traits::test_function_set_type<Test>::value != ObjectType::none,
                        Form<Test, Expr, FormKind::cell>>::type
form(const Test& t, const Expr& e)
{
  return Form<Test, Expr, FormKind::cell>(t, e);
}

template <class Test, class Expr>
typename std::enable_if<Traits::test_function_set_type<Test>::value != ObjectType::none,
                        Form<Test, Expr, FormKind::cell>>::type
form(const Expr& e, const Test& t)
{
  return Form<Test, Expr, FormKind::cell>(t, e);
}

template <class Test, class Expr>
typename std::enable_if<Traits::test_function_set_type<Test>::value != ObjectType::none,
                        Form<Test, Expr, FormKind::face>>::type
face_form(const Test& t, const Expr& e)
{
  return Form<Test, Expr, FormKind::face>(t, e);
}

template <class Test, class Expr>
typename std::enable_if<Traits::test_function_set_type<Test>::value != ObjectType::none,
                        Form<Test, Expr, FormKind::face>>::type
face_form(const Expr& e, const Test& t)
{
  return Form<Test, Expr, FormKind::face>(t, e);
}

template <class Test, class Expr>
typename std::enable_if<Traits::test_function_set_type<Test>::value != ObjectType::none,
                        Form<Test, Expr, FormKind::boundary>>::type
boundary_form(const Test& t, const Expr& e)
{
  return Form<Test, Expr, FormKind::boundary>(t, e);
}

template <class Test, class Expr>
typename std::enable_if<Traits::test_function_set_type<Test>::value != ObjectType::none,
                        Form<Test, Expr, FormKind::boundary>>::type
boundary_form(const Expr& e, const Test& t)
{
  return Form<Test, Expr, FormKind::boundary>(t, e);
}

template <typename... Types>
class Forms;

template <typename FormType>
class Forms<FormType>
{
public:
  static constexpr FormKind form_kind = FormType::form_kind;

  static constexpr bool integrate_value = FormType::integrate_value;
  static constexpr bool integrate_value_exterior =
    (form_kind == FormKind::face) ? FormType::integrate_value_exterior : false;
  static constexpr bool integrate_gradient = FormType::integrate_gradient;
  static constexpr bool integrate_gradient_exterior =
    (form_kind == FormKind::face) ? FormType::integrate_gradient_exterior : false;

  static constexpr unsigned int fe_number = FormType::fe_number;
  static constexpr unsigned int number = 0;

  explicit Forms(const FormType& form_)
    : form(form_)
  {
    AssertThrow(
      (check_forms<number, std::remove_cv_t<decltype(FormType::TestType::integration_flags)>>()),
      dealii::ExcMessage("There are multiple forms that try to submit the same information!"));
    static_assert(Traits::is_form<FormType>::value,
                  "You need to construct this with a Form object!");
  }

  static constexpr void
  get_form_kinds(std::array<bool, 3>& use_objects)
  {
    switch (form_kind)
    {
      case FormKind::cell:
        use_objects[0] = true;
        break;

      case FormKind::face:
        use_objects[1] = true;
        break;

      case FormKind::boundary:
        use_objects[2] = true;
        break;

      default:
        static_assert("Invalid FormKind!");
    }
  }

  template <class FEEvaluation>
  static void
  set_integration_flags(FEEvaluation& phi)
  {
    if constexpr(form_kind == FormKind::cell)
        phi.template set_integration_flags<fe_number>(integrate_value, integrate_gradient);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags_face(FEEvaluation& phi)
  {
    if constexpr(form_kind == FormKind::face)
        phi.template set_integration_flags_face_and_boundary<fe_number>(
          integrate_value,
          integrate_value_exterior,
          integrate_gradient,
          integrate_gradient_exterior);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags_boundary(FEEvaluation& phi)
  {
    if constexpr(form_kind == FormKind::boundary)
        phi.template set_integration_flags_face_and_boundary<fe_number>(
          integrate_value,
          integrate_value_exterior,
          integrate_gradient,
          integrate_gradient_exterior);
  }

  template <class FEEvaluation>
  void
  set_evaluation_flags(FEEvaluation& phi) const
  {
    if constexpr(form_kind == FormKind::cell) form.expr.set_evaluation_flags(phi);
  }

  template <class FEEvaluation>
  void
  set_evaluation_flags_face(FEEvaluation& phi) const
  {
    if constexpr(form_kind == FormKind::face || form_kind == FormKind::boundary)
        form.expr.set_evaluation_flags(phi);
  }

  template <class FEEvaluation>
  void evaluate([[maybe_unused]] FEEvaluation& phi, [[maybe_unused]] unsigned int q) const
  {
    if constexpr(form_kind == FormKind::cell)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "expecting cell value from fe_number " << fe_number << std::endl;
#endif
        const auto value = form.value(phi, q);
#ifdef DEBUG_OUUTPUT
        std::cout << "expecting cell submit from fe_number " << fe_number << std::endl;
#endif
        form.submit(phi, q, value);
      }
  }

  template <class FEEvaluation>
  void evaluate_face([[maybe_unused]] FEEvaluation& phi, [[maybe_unused]] unsigned int q) const
  {
    if constexpr(form_kind == FormKind::face)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "expecting face value from fe_number " << fe_number << std::endl;
#endif
        const auto value = form.value(phi, q);
#ifdef DEBUG_OUUTPUT
        std::cout << "expecting face submit from fe_number " << fe_number << std::endl;
#endif
        form.submit(phi, q, value);
      }
  }

  template <class FEEvaluation>
  void evaluate_boundary([[maybe_unused]] FEEvaluation& phi, [[maybe_unused]] unsigned int q) const
  {
    if constexpr(form_kind == FormKind::boundary)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "expecting face value from fe_number " << fe_number << std::endl;
#endif
        const auto value = form.value(phi, q);
#ifdef DEBUG_OUUTPUT
        std::cout << "expecting face submit from fe_number " << fe_number << std::endl;
#endif
        form.submit(phi, q, value);
      }
  }

  template <class FEEvaluation>
  static void
  integrate(FEEvaluation& phi)
  {
    phi.template integrate<fe_number>(integrate_value, integrate_gradient);
  }

  const FormType&
  get_form() const
  {
    return form;
  }

protected:
  template <unsigned int size, typename IntegrationFlags>
  static /*constexpr*/ bool
  check_forms(
    std::array<std::tuple<FormKind, unsigned int, std::remove_cv_t<IntegrationFlags>>, size>
      container =
        std::array<std::tuple<FormKind, unsigned int, std::remove_cv_t<IntegrationFlags>>, size>{})
  {
    const IntegrationFlags integration_flags = decltype(form.test)::integration_flags;

    for (unsigned int i = number; i < size; ++i)
    {
      const auto& item = container.at(i);
      if (std::get<0>(item) == form_kind && std::get<1>(item) == fe_number &&
          ((std::get<2>(item)) & integration_flags))
        return false;
    }

    return true;
  }

private:
  const /*static constexpr*/ bool valid =
    check_forms<number, decltype(FormType::TestType::integration_flags)>();
  const FormType form;
};

template <typename FormType, typename... Types>
class Forms<FormType, Types...> : public Forms<Types...>
{
public:
  static constexpr FormKind form_kind = FormType::form_kind;

  static constexpr bool integrate_value = FormType::integrate_value;
  static constexpr bool integrate_value_exterior =
    (form_kind == FormKind::face) ? FormType::integrate_value_exterior : false;
  static constexpr bool integrate_gradient = FormType::integrate_gradient;
  static constexpr bool integrate_gradient_exterior =
    (form_kind == FormKind::face) ? FormType::integrate_gradient_exterior : false;

  static constexpr unsigned int fe_number = FormType::fe_number;
  static constexpr unsigned int number = Forms<Types...>::number + 1;

  Forms(const FormType& form_, const Forms<Types...>& old_form)
    : Forms<Types...>(old_form)
    , form(form_)
  {
    AssertThrow(
      valid,
      dealii::ExcMessage("There are multiple forms that try to submit the same information!"));
    static_assert(Traits::is_form<FormType>::value,
                  "You need to construct this with a Form object!");
  }

  explicit Forms(const FormType& form_, const Types&... old_form)
    : Forms<Types...>(old_form...)
    , form(form_)
  {
    AssertThrow(
      valid,
      dealii::ExcMessage("There are multiple forms that try to submit the same information!"));
    static_assert(Traits::is_form<FormType>::value,
                  "You need to construct this with a Form object!");
  }

  template <class Test, class Expr, FormKind kind_of_form>
  Forms<Form<Test, Expr, kind_of_form>, FormType, Types...>
  operator+(const Form<Test, Expr, kind_of_form>& new_form) const
  {
    return Forms<Form<Test, Expr, kind_of_form>, FormType, Types...>(new_form, *this);
  }

  template <class NewForm1, class NewForm2, typename... NewForms>
  auto
  operator+(const Forms<NewForm1, NewForm2, NewForms...>& new_forms) const
  {
    return Forms<NewForm1, FormType, Types...>(new_forms.get_form(), *this) +
           Forms<NewForm2, NewForms...>(
             static_cast<const Forms<NewForm2, NewForms...>&>(new_forms));
  }

  template <class NewForm>
  auto
  operator+(const Forms<NewForm>& new_forms) const
  {
    return Forms<NewForm, FormType, Types...>(new_forms.get_form(), *this);
  }

  static constexpr void
  get_form_kinds(std::array<bool, 3>& use_objects)
  {
    switch (form_kind)
    {
      case FormKind::cell:
        use_objects[0] = true;
        break;

      case FormKind::face:
        use_objects[1] = true;
        break;

      case FormKind::boundary:
        use_objects[2] = true;
        break;

      default:
        static_assert("Invalid FormKind!");
    }
    Forms<Types...>::get_form_kinds(use_objects);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags(FEEvaluation& phi)
  {
    if constexpr(form_kind == FormKind::cell)
        phi.template set_integration_flags<fe_number>(integrate_value, integrate_gradient);
    Forms<Types...>::set_integration_flags(phi);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags_face(FEEvaluation& phi)
  {
    if constexpr(form_kind == FormKind::face)
        phi.template set_integration_flags_face_and_boundary<fe_number>(
          integrate_value,
          integrate_value_exterior,
          integrate_gradient,
          integrate_gradient_exterior);
    Forms<Types...>::set_integration_flags_face(phi);
  }

  template <class FEEvaluation>
  static void
  set_integration_flags_boundary(FEEvaluation& phi)
  {
    if constexpr(form_kind == FormKind::boundary)
        phi.template set_integration_flags_face_and_boundary<fe_number>(
          integrate_value,
          integrate_value_exterior,
          integrate_gradient,
          integrate_gradient_exterior);
    Forms<Types...>::set_integration_flags_boundary(phi);
  }

  template <class FEEvaluation>
  void
  set_evaluation_flags(FEEvaluation& phi) const
  {
    if constexpr(form_kind == FormKind::cell) form.expr.set_evaluation_flags(phi);
    Forms<Types...>::set_evaluation_flags(phi);
  }

  template <class FEEvaluation>
  void
  set_evaluation_flags_face(FEEvaluation& phi) const
  {
    if constexpr(form_kind == FormKind::face || form_kind == FormKind::boundary)
        form.expr.set_evaluation_flags(phi);
    Forms<Types...>::set_evaluation_flags_face(phi);
  }

  template <class FEEvaluation>
  void
  evaluate(FEEvaluation& phi, unsigned int q) const
  {
    if constexpr(form_kind == FormKind::cell)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "expecting cell value from fe_number " << fe_number << std::endl;
#endif
        const auto value = form.value(phi, q);
#ifdef DEBUG_OUTPUT
        std::cout << "descending" << std::endl;
#endif
        Forms<Types...>::evaluate(phi, q);
#ifdef DEBUG_OUTPUT
        std::cout << "expecting cell submit from fe_number " << fe_number << std::endl;
#endif
        form.submit(phi, q, value);
      }
    else
      Forms<Types...>::evaluate(phi, q);
  }

  template <class FEEvaluation>
  void
  evaluate_face(FEEvaluation& phi, unsigned int q) const
  {
    if constexpr(form_kind == FormKind::face)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "expecting face value from fe_number " << fe_number << std::endl;
#endif
        const auto value = form.value(phi, q);
#ifdef DEBUG_OUTPUT
        std::cout << "descending" << std::endl;
#endif
        Forms<Types...>::evaluate_face(phi, q);
#ifdef DEBUG_OUTPUT
        std::cout << "expecting face submit from fe_number " << fe_number << std::endl;
#endif
        form.submit(phi, q, value);
      }
    else
      Forms<Types...>::evaluate_face(phi, q);
  }

  template <class FEEvaluation>
  void
  evaluate_boundary(FEEvaluation& phi, unsigned int q) const
  {
    if constexpr(form_kind == FormKind::boundary)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "expecting face value from fe_number " << fe_number << std::endl;
#endif
        const auto value = form.value(phi, q);
#ifdef DEBUG_OUTPUT
        std::cout << "descending" << std::endl;
#endif
        Forms<Types...>::evaluate_boundary(phi, q);
#ifdef DEBUG_OUTPUT
        std::cout << "expecting face submit from fe_number " << fe_number << std::endl;
#endif
        form.submit(phi, q, value);
      }
    else
      Forms<Types...>::evaluate_boundary(phi, q);
  }

  template <class FEEvaluation>
  static void
  integrate(FEEvaluation& phi)
  {
    phi.template integrate<fe_number>(integrate_value, integrate_gradient);
    Forms<Types...>::integrate(phi);
  }

  const FormType&
  get_form() const
  {
    return form;
  }

  auto operator*(const double scalar) const
  {
    const typename std::remove_reference<decltype(*this)>::type newform(
      form * scalar, Forms<Types...>::get_form() * scalar);
    return newform;
  }

  auto
  operator-() const
  {
    return (*this) * -1.;
  }

protected:
  template <unsigned int size, typename IntegrationFlags>
  static /*constexpr*/ bool
  check_forms(
    std::array<std::tuple<FormKind, unsigned int, std::remove_cv_t<IntegrationFlags>>, size>
      container =
        std::array<std::tuple<FormKind, unsigned int, std::remove_cv_t<IntegrationFlags>>, size>{})
  {
    IntegrationFlags integration_flags = decltype(form.test)::integration_flags;
    const auto new_tuple = std::make_tuple(form_kind, fe_number, integration_flags);

    for (unsigned int i = number; i < size; ++i)
    {
      const auto& item = container.at(i);
      if (std::get<0>(item) == form_kind && std::get<1>(item) == fe_number &&
          ((std::get<2>(item)) & integration_flags))
        return false;
    }

    container[number - 1] = new_tuple;
    return Forms<Types...>::template check_forms<size, IntegrationFlags>(container);
  }

private:
  const /*static constexpr*/ bool valid =
    check_forms<number, std::remove_cv_t<decltype(FormType::TestType::integration_flags)>>();
  const FormType form;
};

template <class... Args>
struct CFL::Traits::is_multiplicable<Forms<Args...>, double>
{
  static const bool value = true;
};

template <class... Args>
auto operator*(double scalar, const Forms<Args...>& forms)
{
  return forms * scalar;
}

template <class... Args>
struct CFL::Traits::is_multiplicable<Form<Args...>, double>
{
  static const bool value = true;
};

template <class Test, class Expr, FormKind kind_of_form, typename NumberType>
auto operator*(double scalar, const Form<Test, Expr, kind_of_form, NumberType>& form)
{
  return form * scalar;
}

} // namespace CFL

#endif