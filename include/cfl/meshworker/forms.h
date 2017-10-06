#ifndef cfl_forms_h
#define cfl_forms_h

#include <array>
#include <iostream>
#include <string>
#include <utility>

#include <tuple>

#include <cfl/base/traits.h>
#include <cfl/base/constants.h>

namespace CFL{
  namespace dealii::MeshWorker {
  template<int rank, class Test, class Expr>
  struct form_latex_aux {
    std::string operator()(const Test & /*test*/, const Expr & /*expr*/) {
      return std::string("Not implemented for rank ") + std::to_string(rank);
    }
  };

  template<class Test, class Expr>
  struct form_latex_aux<0, Test, Expr> {
    std::string operator()(const Test &test, const Expr &expr) {
      return R"(\left()" + expr.latex() + "," + test.latex() + R"(\right))";
    }
  };

  template<class Test, class Expr>
  struct form_latex_aux<1, Test, Expr> {
    std::string operator()(const Test &test, const Expr &expr) {
      std::string output;
      for (unsigned int i = 0; i < Test::TensorTraits::dim; ++i) {
        if (i > 0) {
          output += " + ";
          output += R"(\left()" + expr.latex(i) + "," + test.latex(i) + R"(\right))";
        }
        return output;
      }
    }
  };

  template<class Test, class Expr>
  struct form_latex_aux<2, Test, Expr> {
    std::string operator()(const Test &test, const Expr &expr) {
      std::string output;
      for (unsigned int i = 0; i < Test::TensorTraits::dim; ++i) {
        for (unsigned int j = 0; j < Test::TensorTraits::dim; ++j) {
          if (i > 0 || j > 0)
            output += " + ";
          output += R"(\left()" + expr.latex(i, j) + "," + test.latex(i, j) + R"(\right))";
        }
      }
      return output;
    }
  };

  template<int rank, class Test, class Expr>
  struct form_evaluate_aux {
    double operator()(unsigned int /*k*/, unsigned int /*i*/, const Test & /*test*/, const Expr & /*expr*/) {
      static_assert(rank < 2, "Not implemented for this rank");
      return 0.;
    }
  };

  template<class Test, class Expr>
  struct form_evaluate_aux<0, Test, Expr> {
    double operator()(unsigned int k, unsigned int i, const Test &test, const Expr &expr) {
      return test.evaluate(k, i) * expr.evaluate(k);
    }
  };

  template<class Test, class Expr>
  struct form_evaluate_aux<1, Test, Expr> {
    double operator()(unsigned int k, unsigned int i, const Test &test, const Expr &expr) {
      double sum = 0.;
      for (unsigned int d = 0; d < Test::TensorTraits::dim; ++d) {
        sum += test.evaluate(k, i, d) * expr.evaluate(k, d);
      }
      return sum;
    }
  };

  namespace {
    enum class FormKind {
      cell, face, boundary
    };
  }

  template<FormKind, ObjectType>
  constexpr bool formkind_matches_objecttype() {
    return false;
  }

  template<>
  constexpr bool formkind_matches_objecttype<FormKind::cell, ObjectType::cell>() {
    return true;
  }

  template<>
  constexpr bool formkind_matches_objecttype<FormKind::face, ObjectType::face>() {
    return true;
  }

  template<>
  constexpr bool formkind_matches_objecttype<FormKind::boundary, ObjectType::face>() {
    return true;
  }

  template<typename... Types>
  class Forms;

/**
 * A Form is an expression tested by a test function set.
 */
  template<class Test, class Expr, FormKind kind_of_form, typename NumberType = double>
  class Form final {
    public:
    using TestType = Test;
    const Test test_data;
    const Expr expr_data;

    static constexpr FormKind form_kind = kind_of_form;

    constexpr Form(Test test_, Expr expr_) : test_data(std::move(test_)), expr_data(std::move(expr_)) {
      static_assert(Traits::test_function_set_type<Test>::value != ObjectType::none,
                    "The first argument must be a test function!");
      static_assert(Traits::fe_function_set_type<Expr>::value != ObjectType::none,
                    "The second argument must be a finite element function!");
      static_assert(formkind_matches_objecttype<kind_of_form, Traits::test_function_set_type<Test>::value>(),
                    "The type of the test function must be compatible with the type of the form!");
      static_assert(formkind_matches_objecttype<kind_of_form, Traits::fe_function_set_type<Expr>::value>(),
                    "The type of the expression must be compatible with the type of the form!");
      static_assert(Test::TensorTraits::rank == Expr::TensorTraits::rank,
                    "Test function and expression must have the same tensor rank!");
      static_assert(Test::TensorTraits::dim == Expr::TensorTraits::dim,
                    "Test function and expression must have the same dimension!");

      if constexpr(form_kind == FormKind::boundary) static_assert(
        Test::integration_flags.value_exterior == false && Test::integration_flags.gradient_exterior == false,
        "A boundary form cannot have a test function associated with the neighbor of a cell!");
    }


    constexpr const Expr &expr() const {
      return expr_data;
    }

    constexpr const Test &test() const {
      return test_data;
    }

    static constexpr void get_form_kinds(std::array<bool, 3> &use_objects) {
      switch (form_kind) {
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

    std::string latex() const {
      return form_latex_aux<Test::TensorTraits::rank, Test, Expr>()(test_data, expr_data);
    }

    template<class TestNew, class ExprNew, FormKind kind_new>
    Forms<Form<Test, Expr, kind_of_form>, Form<TestNew, ExprNew, kind_new>>
    operator+(const Form<TestNew, ExprNew, kind_new> &new_form) const {
      return Forms<Form<Test, Expr, kind_of_form>, Form<TestNew, ExprNew, kind_new>>(*this, new_form);
    }

    template<class TestNew, class ExprNew, FormKind kind_new>
    auto operator-(const Form<TestNew, ExprNew, kind_new> &new_form) const {
      return Forms<Form<Test, Expr, kind_of_form>, Form<TestNew, ConstantScaled<ExprNew, NumberType>, kind_new>>(*this,
                                                                                                                 -new_form);
    }

    template<class... Types>
    auto operator+(const Forms<Types...> &old_form) const {
      return old_form + *this;
    }

    auto operator-() const {
      const Form<Test, ConstantScaled<Expr, NumberType>, kind_of_form, NumberType> newform(test_data,
                                                                                           scale(-1., expr_data));
      return newform;
    }

    auto operator*(const double scalar) const {
      const Form<Test, ConstantScaled<Expr, NumberType>, kind_of_form, NumberType> newform(test_data,
                                                                                           scale(scalar, expr_data));
      return newform;
    }
  };
}

namespace Traits
{
  template <class Test, class Expr, dealii::MeshWorker::FormKind kind_of_form>
  struct is_form<dealii::MeshWorker::Form<Test, Expr, kind_of_form>>
  {
    const static bool value = true;
  };

  template <class Test, class Expr, dealii::MeshWorker::FormKind kind_of_form>
  struct is_cfl_object<dealii::MeshWorker::Form<Test, Expr, kind_of_form>>
  {
    const static bool value = true;
  };

  template <class Test1, class Expr1, dealii::MeshWorker::FormKind kind1, class Test2, class Expr2, dealii::MeshWorker::FormKind kind2>
  struct is_summable<dealii::MeshWorker::Form<Test1, Expr1, kind1>, dealii::MeshWorker::Form<Test2, Expr2, kind2>>
  {
    const static bool value = true;
  };

  template <class Test, class Expr, dealii::MeshWorker::FormKind kind_of_form, typename... Types>
  struct is_summable<dealii::MeshWorker::Form<Test, Expr, kind_of_form>, dealii::MeshWorker::Forms<Types...>>
  {
    const static bool value = true;
  };

  template <class Test, class Expr, dealii::MeshWorker::FormKind kind_of_form, typename... Types>
  struct is_summable<dealii::MeshWorker::Forms<Types...>, dealii::MeshWorker::Form<Test, Expr, kind_of_form>>
  {
    const static bool value = true;
  };
} // namespace Traits

namespace dealii::MeshWorker {

  template<class Test, class Expr>
  typename std::enable_if<
    Traits::test_function_set_type<Test>::value != ObjectType::none, Form<Test, Expr, FormKind::cell>>::type
  form(const Test &t, const Expr &e) {
    return Form<Test, Expr, FormKind::cell>(t, e);
  }

  template<class Test, class Expr>
  typename std::enable_if<
    Traits::test_function_set_type<Test>::value != ObjectType::none, Form<Test, Expr, FormKind::cell>>::type
  form(const Expr &e, const Test &t) {
    return Form<Test, Expr, FormKind::cell>(t, e);
  }

  template<class Test, class Expr>
  typename std::enable_if<
    Traits::test_function_set_type<Test>::value != ObjectType::none, Form<Test, Expr, FormKind::face>>

  ::type face_form(const Test &t, const Expr &e) {
    return Form<Test, Expr, FormKind::face>(t, e);
  }

  template<class Test, class Expr>
  typename std::enable_if<
    Traits::test_function_set_type<Test>::value != ObjectType::none, Form<Test, Expr, FormKind::face>>

  ::type face_form(const Expr &e, const Test &t) {
    return Form<Test, Expr, FormKind::face>(t, e);
  }

  template<class Test, class Expr>
  typename std::enable_if<
    Traits::test_function_set_type<Test>::value != ObjectType::none, Form<Test, Expr, FormKind::boundary>>

  ::type boundary_form(const Test &t, const Expr &e) {
    return Form<Test, Expr, FormKind::boundary>(t, e);
  }

  template<class Test, class Expr>
  typename std::enable_if<
    Traits::test_function_set_type<Test>::value != ObjectType::none, Form<Test, Expr, FormKind::boundary>>

  ::type boundary_form(const Expr &e, const Test &t) {
    return Form<Test, Expr, FormKind::boundary>(t, e);
  }

  template<typename... Types>
  class Forms;

  template<typename FormType>
  class Forms<FormType> {
    public:
    static constexpr FormKind form_kind = FormType::form_kind;
    static constexpr unsigned int number = 0;

    explicit Forms(const FormType &form_) : form(form_) {
      static_assert(Traits::is_form<FormType>::value, "You need to construct this with a Form object!");
    }

    static constexpr void get_form_kinds(std::array<bool, 3> &use_objects) {
      switch (form_kind) {
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

    const FormType &get_form() const {
      return form;
    }

    private:
    const FormType form;
  };

  template<typename FormType, typename... Types>
  class Forms<FormType, Types...> : public Forms<Types...> {
    public:
    static constexpr FormKind form_kind = FormType::form_kind;
    static constexpr unsigned int number = Forms<Types...>::number + 1;

    Forms(const FormType &form_, const Forms<Types...> &old_form) : Forms<Types...>(old_form), form(form_) {
      static_assert(Traits::is_form<FormType>::value, "You need to construct this with a Form object!");
    }

    explicit Forms(const FormType &form_, const Types &... old_form) : Forms<Types...>(old_form...), form(form_) {
      static_assert(Traits::is_form<FormType>::value, "You need to construct this with a Form object!");
    }

    template<class Test, class Expr, FormKind kind_of_form>
    Forms<Form<Test, Expr, kind_of_form>, FormType, Types...>

    operator+(const Form<Test, Expr, kind_of_form> &new_form) const {
      return Forms<Form<Test, Expr, kind_of_form>, FormType, Types...>(new_form, *this);
    }

    template<class NewForm1, class NewForm2, typename... NewForms>
    auto operator+(const Forms<NewForm1, NewForm2, NewForms...> &new_forms) const {
      return Forms<NewForm1, FormType, Types...>(new_forms.get_form(), *this) +
             Forms<NewForm2, NewForms...>(static_cast<const Forms<NewForm2, NewForms...> &>(new_forms));
    }

    template<class NewForm>
    auto operator+(const Forms<NewForm> &new_forms) const {
      return Forms<NewForm, FormType, Types...>(new_forms.get_form(), *this);
    }

    static constexpr void get_form_kinds(std::array<bool, 3> &use_objects) {
      switch (form_kind) {
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

    const Forms<Types...> &get_other() const {
      return *this;
    }

    const FormType &get_form() const {
      return form;
    }

    auto operator*(const double scalar) const {
      const typename std::remove_reference<decltype(*this)>::type newform(form * scalar,
                                                                          Forms<Types...>::get_form() * scalar);
      return newform;
    }

    auto operator-() const {
      return (*this) * -1.;
    }

    private:
    const FormType form;
  };
}

  template<class... Args>
  struct CFL::Traits::is_multiplicable<dealii::MeshWorker::Forms<Args...>, double> {
    static const bool value = true;
  };

  template<class... Args>
  auto operator*(double scalar, const dealii::MeshWorker::Forms<Args...> &forms) {
    return forms * scalar;
  }

  template<class... Args>
  struct CFL::Traits::is_multiplicable<dealii::MeshWorker::Form < Args...>, double> {
  static const bool value = true;
};

template<class Test, class Expr, dealii::MeshWorker::FormKind kind_of_form, typename NumberType>
auto operator*(double scalar, const dealii::MeshWorker::Form <Test, Expr, kind_of_form, NumberType> &form) {
  return form * scalar;
}

} // namespace CFL

#endif
