#ifndef cfl_dealii_matrixfree_forms_h
#define cfl_dealii_matrixfree_forms_h

#include <array>
#include <iostream>
#include <string>
#include <utility>

#include <tuple>

#include <deal.II/base/exceptions.h>

#include <cfl/forms.h>
#include <cfl/traits.h>

#include <matrixfree/fefunctions.h>

namespace CFL
{
namespace dealii::MatrixFree
{
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

    template <class OtherTest, class OtherExpr>
    constexpr Form(const Base::Form<OtherTest, OtherExpr, kind_of_form, NumberType> f)
      : test(transform(f.test))
      , expr(transform(f.expr))
    {
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
    void evaluate_boundary([[maybe_unused]] FEEvaluation& phi,
                           [[maybe_unused]] unsigned int q) const
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
  };

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

    template <class OtherTest, class OtherExpr, typename NumberType>
    constexpr Forms(const Base::Forms<Base::Form<OtherTest, OtherExpr, form_kind, NumberType>>& f)
      : form(f.get_form())
    {
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
    void evaluate_boundary([[maybe_unused]] FEEvaluation& phi,
                           [[maybe_unused]] unsigned int q) const
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

  private:
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

    template <class OtherType, class... OtherTypes,
              typename std::enable_if<sizeof...(OtherTypes) == sizeof...(Types)>::type* = nullptr>
    constexpr Forms(const Base::Forms<OtherType, OtherTypes...>& f)
      : Forms<Types...>(static_cast<Base::Forms<OtherTypes...>>(f))
      , form(f.get_form())
    {
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

  private:
    const FormType form;
  };

  template <class Test, class Expr, FormKind kind_of_form, typename NumberType>
  constexpr auto
  transform(const Base::Form<Test, Expr, kind_of_form, NumberType>& f)
  {
    return Form<decltype(transform(std::declval<Test>())),
                decltype(transform(std::declval<Expr>())),
                kind_of_form>(f);
  }

  template <typename... Types>
  constexpr auto
  transform(const Base::Forms<Types...>& f)
  {
    return Forms<decltype(transform(std::declval<Types>()))...>(f);
  }
}
} // namespace CFL

#endif
