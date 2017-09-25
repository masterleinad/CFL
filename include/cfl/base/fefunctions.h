#ifndef cfl_fefunctions_h
#define cfl_fefunctions_h

#include <cfl/base/forms.h>
#include <cfl/base/traits.h>

#include <utility>

namespace CFL
{
namespace Base
{
  template <class Derived>
  class TestFunctionBaseBase;

  template <class Derived>
  class TestFunctionBase;

  template <class Derived>
  class TestFunctionFaceBase;

  template <class Derived>
  class FEFunctionBaseBase;

  template <class Derived>
  class FEFunctionBase;

  template <class Derived>
  class FEFunctionFaceBase;

  template <typename... Types>
  class SumFEFunctions;

  template <typename... Types>
  class ProductFEFunctions;

  template <class FEFunctionType>
  class FELiftDivergence;

  namespace internal::optimize
  {
    template <std::size_t... Ns>
    struct sequence
    {
    };
    template <std::size_t... Ns>
    struct seq_gen;

    template <std::size_t first, std::size_t second, std::size_t... Ns>
    struct seq_gen<first, second, Ns...>
    {
      static_assert(first <= second, "The end must not be less than the start!");
      using type1 = sequence<first, second, Ns...>;
      using type2 = typename seq_gen<first, second - 1, second, Ns...>::type;

      using cond1 = typename std::conditional<first == second - 1, type1, type2>::type;
      using type = typename std::conditional<first == second, sequence<first>, cond1>::type;
    };

    template <std::size_t first, std::size_t... Ns>
    struct seq_gen<first, 0, Ns...>
    {
      using type = sequence<0>;
    };

    template <std::size_t start, std::size_t end>
    using sequence_t = typename std::conditional<(start > end), sequence<>,
                                                 typename seq_gen<start, end>::type>::type;

    // allow for appending an object to an std::array object.
    template <typename T, std::size_t N, std::size_t... I>
    constexpr std::array<T, N + 1>
    append_aux(std::array<T, N> a, T t, sequence<I...>)
    {
      return std::array<T, N + 1>{ { a[I]..., t } };
    }

    template <typename T, std::size_t N1, std::size_t... I1, std::size_t N2, std::size_t... I2>
    constexpr std::array<T, N1 + N2>
    append_aux(std::array<T, N1> a1, sequence<I1...>, std::array<T, N2> a2, sequence<I1...>)
    {
      return std::array<T, N1 + N2>{ { a1[I1]..., a2[I2]... } };
    }

    template <typename T, std::size_t N>
    constexpr std::array<T, N + 1>
    append(std::array<T, N> a, T t)
    {
      if constexpr(N == 0)
        {
          (void)a;
          return std::array<T, 1>{ { t } };
        }
      else
        return append_aux(a, t, sequence_t<0, N - 1>());
    }

    template <class Type, class... Types>
    constexpr std::tuple<Types..., Type>
    append(const std::tuple<Types...>& tuple, const Type& t)
    {
      return std::tuple_cat(tuple, std::make_tuple(t));
    }

    template <typename T, std::size_t N, std::size_t... I>
    constexpr std::array<T, sizeof...(I)>
    extract(std::array<T, N> a, sequence<I...>)
    {
      return std::array<T, sizeof...(I)>{ { a[I]... } };
    }

    template <class... Types>
    class TypeStorage
    {
    };

    template <typename T, std::size_t N1, std::size_t N2>
    constexpr std::array<T, N1 + N2>
    append(std::array<T, N1> a1, std::array<T, N2> a2)
    {
      return append_aux(a1, sequence_t<0, N1 - 1>(), a2, sequence_t<0, N2 - 1>());
    }

    template <int n, class Type, class... StorageTypes>
    struct TypeExists;

    template <int n, class Type, class FirstStorageType, class... StorageTypes>
    struct TypeExists<n, Type, FirstStorageType, StorageTypes...>
    {
      using type =
        typename std::conditional<std::is_same<Type, FirstStorageType>::value,
                                  std::pair<std::true_type, std::integral_constant<int, n>>,
                                  typename TypeExists<n + 1, Type, StorageTypes...>::type>::type;
      static constexpr bool value = decltype(std::declval<type>().first)::value;
      static constexpr unsigned int position = decltype(std::declval<type>().second)::value;
    };

    template <int n, class Type>
    struct TypeExists<n, Type>
    {
      using type = std::pair<std::false_type, std::integral_constant<int, n>>;
      static constexpr bool value = false;
      static constexpr unsigned int position = n;
    };

    template <class Type, class... StorageTypes>
    inline constexpr auto
    create_list(const SumFEFunctions<Type>& sum, std::tuple<StorageTypes...> storage)
    {
      using ResultType = TypeExists<0, Type, StorageTypes...>;
      if constexpr(ResultType::value)
        {
          constexpr unsigned int i = ResultType::position;
          std::get<i>(storage) += sum.get_summand();
          return storage;
        }
      else
      {
        // no, there doesn't exist such an object yet.
        auto new_storage = append(storage, sum.get_summand());
        return new_storage;
      }
    }

    template <class Type, class... Types, class... StorageTypes>
    constexpr auto
    create_list(SumFEFunctions<Type, Types...> sum, std::tuple<StorageTypes...> storage,
                typename std::enable_if<sizeof...(Types) != 0>::type* = nullptr)
    {
      using ResultType = TypeExists<0, Type, StorageTypes...>;
      if constexpr(ResultType::value)
        {
          constexpr unsigned int i = ResultType::position;
          std::get<i>(storage) += sum.get_summand();
          return create_list(static_cast<SumFEFunctions<Types...>>(sum), storage);
        }
      else
      {
        // no, there doesn't exist such an object yet.
        auto new_storage = append(storage, sum.get_summand());
        return create_list(static_cast<SumFEFunctions<Types...>>(sum), new_storage);
      }
    }

    template <int i, class StorageType>
    inline constexpr auto
    create_types(std::pair<TypeStorage<StorageType>, std::array<double, 1>> storage)
    {
      return StorageType(storage.second[0]);
    }

    template <int i, class... StorageTypes>
    inline constexpr auto
    create_types(std::tuple<StorageTypes...> storage)
    {
      auto function = std::get<i>(storage);
      if constexpr(i < sizeof...(StorageTypes) - 1)
        return function + create_types<i + 1>(storage);
      else
      return function;
    }
  }
}

namespace Traits
{
  /**
   * @brief Trait to determine if a given type is CFL SumFEFunctions
   *
   */
  template <typename... Types>
  struct is_cfl_object<Base::SumFEFunctions<Types...>>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief Trait to determine if a given type is CFL object
   *
   * Trait to determine if a given type is derived from CFL \ref TestFunctionBaseBase
   *
   */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct is_cfl_object<T<rank, dim, idx>,
                       std::enable_if_t<std::is_base_of<
                         Base::TestFunctionBaseBase<T<rank, dim, idx>>, T<rank, dim, idx>>::value>>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief Trait to determine if a given type is CFL object
   *
   * Trait to determine if a given type is derived from CFL \ref FELiftDivergence
   *
   */
  template <class FEFunctionType>
  struct is_cfl_object<Base::FELiftDivergence<FEFunctionType>>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief Trait to determine if a given type is CFL object
   *
   * Trait to determine if a given type is derived from CFL \ref FEFunctionBaseBase
   *
   */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct is_cfl_object<T<rank, dim, idx>,
                       std::enable_if_t<std::is_base_of<Base::FEFunctionBaseBase<T<rank, dim, idx>>,
                                                        T<rank, dim, idx>>::value>>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief Trait to determine if two FE functions can be summed together
   *
   * This trait is used to check if two FE functions are of same type. i.e.
   * (cell/face) \ref ObjectType so that they can be summed together
   *
   */
  template <typename A, typename B>
  struct is_summable<
    A, B,
    typename std::enable_if<fe_function_set_type<A>::value != ObjectType::none &&
                            fe_function_set_type<A>::value == fe_function_set_type<B>::value>::type>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief Trait to determine if two FE functions can be multiplied together
   *
   * This trait is used to check if two FE functions are of same type. i.e.
   * (cell/face) \ref ObjectType so that they can be multiplied together
   *
   */
  template <typename A, typename B>
  struct is_multiplicable<
    A, B,
    typename std::enable_if<fe_function_set_type<A>::value != ObjectType::none &&
                            fe_function_set_type<A>::value == fe_function_set_type<B>::value>::type>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief Trait to determine if a given constant can be multipled with FE function
   *
   * This trait is used to check if a given constant type is arithmetic so that
   * it can be multiplied with FE function
   *
   */
  template <typename A, typename B>
  struct is_multiplicable<
    A, B, typename std::enable_if_t<std::is_arithmetic<A>() &&
                                    fe_function_set_type<B>::value != ObjectType::none>>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief Trait to determine if a given constant can be multipled with FE function
   *
   * This trait is used to check if a given constant type is arithmetic so that
   * it can be multiplied with FE function
   *
   */
  template <typename A, typename B>
  struct is_multiplicable<
    A, B, typename std::enable_if_t<fe_function_set_type<A>::value != ObjectType::none &&
                                    std::is_arithmetic<B>()>>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief Trait to store measure region as cell for a test function
   *
   * This trait is used to check if the given test function is derived from CFL
   * \ref TestFunctionBase and marks its \ref ObjectType as cell
   *
   */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct test_function_set_type<
    T<rank, dim, idx>, std::enable_if_t<std::is_base_of<Base::TestFunctionBase<T<rank, dim, idx>>,
                                                        T<rank, dim, idx>>::value>>
  {
    static constexpr ObjectType value = ObjectType::cell;
  };

  /**
   * @brief Trait to store measure region as face for a test function
   *
   * This trait is used to check if the given test function is derived from CFL
   * \ref TestFunctionFaceBase and marks its \ref ObjectType as face
   *
   */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct test_function_set_type<
    T<rank, dim, idx>, std::enable_if_t<std::is_base_of<
                         Base::TestFunctionFaceBase<T<rank, dim, idx>>, T<rank, dim, idx>>::value>>
  {
    static constexpr ObjectType value = ObjectType::face;
  };

  /**
   * @brief Trait to store measure region as cell for a \ref FELiftDivergence function
   *
   * This trait is used to check if the given FE function is of type CFL
   * \ref FELiftDivergence and marks its \ref ObjectType as cell
   *
   */
  template <class FEFunctionType>
  struct fe_function_set_type<Base::FELiftDivergence<FEFunctionType>>
  {
    static constexpr ObjectType value = ObjectType::cell;
  };

  /**
   * @brief Trait to store measure region as cell type for a FE function
   *
   * This trait is used to check if the given FE function is derived from CFL
   * \ref FEFunctionBase and marks its \ref ObjectType as cell
   *
   */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct fe_function_set_type<
    T<rank, dim, idx>, std::enable_if_t<std::is_base_of<Base::FEFunctionBase<T<rank, dim, idx>>,
                                                        T<rank, dim, idx>>::value>>
  {
    static constexpr ObjectType value = ObjectType::cell;
  };

  /**
   * @brief Trait to store measure region as face for a FE function
   *
   * This trait is used to check if the given FE function is derived from CFL
   * \ref FEFunctionFaceBase and marks its \ref ObjectType as face
   *
   */
  template <template <int, int, unsigned int> class T, int rank, int dim, unsigned int idx>
  struct fe_function_set_type<
    T<rank, dim, idx>, std::enable_if_t<std::is_base_of<Base::FEFunctionFaceBase<T<rank, dim, idx>>,
                                                        T<rank, dim, idx>>::value>>
  {
    static constexpr ObjectType value = ObjectType::face;
  };

  /**
   * @brief Trait to store measure region as face for a \ref SumFEFunctions function
   *
   * This trait is used to mark the \ref ObjectType of an object of type CFL
   * \ref SumFEFunctions as the measure region of its first constituting element
   *
   */
  template <class FirstType, typename... Types>
  struct fe_function_set_type<Base::SumFEFunctions<FirstType, Types...>>
  {
    static constexpr ObjectType value = fe_function_set_type<FirstType>::value;
  };

  /**
   * @brief Trait to store measure region as face for a \ref ProductFEFunctions function
   *
   * This trait is used to mark the \ref ObjectType of an object of type CFL
   * \ref ProductFEFunctions as the measure region of its first constituting element
   *
   */
  template <class FirstType, typename... Types>
  struct fe_function_set_type<Base::ProductFEFunctions<FirstType, Types...>>
  {
    static constexpr ObjectType value = fe_function_set_type<FirstType>::value;
  };

  /**
   * @brief Determine if a given object is of type CFL \ref ProductFEFunctions
   *
   * Default Trait
   * @todo Should it be removed? Discuss
   */
  template <class T>
  struct is_fe_function_product
  {
    static constexpr bool value = false;
  };

  /**
   * @brief Determine if a given object is of type CFL \ref ProductFEFunctions
   *
   */
  template <typename... Types>
  struct is_fe_function_product<Base::ProductFEFunctions<Types...>>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief Determine if a given object is of type CFL \ref SumFEFunctions
   *
   * Default Trait
   * @todo Should it be removed? Discuss
   */
  template <class T>
  struct is_fe_function_sum
  {
    static constexpr bool value = false;
  };

  /**
   * @brief Determine if a given object is of type CFL \ref SumFEFunctions
   *
   */
  template <typename... Types>
  struct is_fe_function_sum<Base::SumFEFunctions<Types...>>
  {
    static constexpr bool value = true;
  };
} // namespace Traits

namespace Base
{
  /**
   * @brief TBD
   * \todo Add details
   *
   */
  struct IntegrationFlags
  {
    bool value = false;
    bool value_exterior = false;
    bool gradient = false;
    bool gradient_exterior = false;

    constexpr bool operator&(const IntegrationFlags& other_flags) const
    {
      return (value & other_flags.value) || (value_exterior & other_flags.value_exterior) ||
             (gradient & other_flags.gradient) ||
             (gradient_exterior & other_flags.gradient_exterior);
    }
  };

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
    static constexpr const IntegrationFlags integration_flags{ true, false, false, false };
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
    static constexpr IntegrationFlags integration_flags{ false, true, false, false };
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
    static constexpr IntegrationFlags integration_flags{ false, false, true, false };
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
    static constexpr IntegrationFlags integration_flags{ false, false, false, true };
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
    static constexpr IntegrationFlags integration_flags{ true, false, false, false };
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
    static constexpr IntegrationFlags integration_flags{ false, false, true, false };
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
    static constexpr IntegrationFlags integration_flags{ false, false, true, false };
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
    static constexpr IntegrationFlags integration_flags{ false, false, true, false };
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
    static constexpr IntegrationFlags integration_flags{ false, false, true, false };
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
    static constexpr IntegrationFlags integration_flags{ false, false, false, false };
  };

  /**
   * Utility function to return a TestDivergence object
   * given a TestFunction
   *
   */
  template <int rank, int dim, unsigned int idx>
  TestDivergence<rank - 1, dim, idx>
  div(const TestFunction<rank, dim, idx>& /*unused*/)
  {
    return TestDivergence<rank - 1, dim, idx>();
  }

  /**
   * Utility function to return a TestGradient object
   * given a TestFunction
   *
   */
  template <int rank, int dim, unsigned int idx>
  TestGradient<rank + 1, dim, idx> constexpr grad(const TestFunction<rank, dim, idx>& /*unused*/)
  {
    return TestGradient<rank + 1, dim, idx>();
  }

  /**
   * Utility function to return a TestHessian object
   * given a TestGradient
   *
   */
  template <int rank, int dim, unsigned int idx>
  TestHessian<rank + 1, dim, idx> constexpr grad(const TestGradient<rank, dim, idx>& /*unused*/)
  {
    return TestHessian<rank + 1, dim, idx>();
  }

  // CRTP

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
  protected:
    /**
     * Default constructor
     *
     */
    explicit constexpr FEFunctionBaseBase(double new_factor = 1.)
      : scalar_factor(new_factor)
    {
    }

  public:
    using TensorTraits = Traits::Tensor<rank, dim>;
    static constexpr unsigned int index = idx;
    double scalar_factor = 1.;

    constexpr void
    operator+=(const Derived<rank, dim, idx>& other_function)
    {
      scalar_factor += other_function.scalar_factor;
    };

    /**
     * Allows to scale an FE function with a arithmetic value
     *
     */
    template <typename Number>
    constexpr typename std::enable_if_t<std::is_arithmetic<Number>::value, Derived<rank, dim, idx>>
    operator*(const Number scalar_factor_) const
    {
      return Derived<rank, dim, idx>(scalar_factor * scalar_factor_);
    }

    /**
     * Allows to negate an FE function
     *
     */
    constexpr Derived<rank, dim, idx>
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
  protected:
    using FEFunctionBaseBase<Derived>::FEFunctionBaseBase;
    using FEFunctionBaseBase<Derived>::operator+=;
  };

  /**
   * Top level base class for FE Functions on Face
   * See \ref TestFunctionBase for more details
   *
   */
  template <class Derived>
  class FEFunctionFaceBase : public FEFunctionBaseBase<Derived>
  {
  protected:
    using FEFunctionBaseBase<Derived>::FEFunctionBaseBase;
    using FEFunctionBaseBase<Derived>::operator+=;
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
    using Base::operator+=;
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
    using Base::operator+=;
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
    using Base::operator+=;
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
    using Base::operator+=;
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
    using Base::operator+=;
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
    using Base::operator+=;

    explicit constexpr FEDivergence(const FEFunction<rank + 1, dim, idx>& fefunction)
      : FEDivergence(fefunction.scalar_factor)
    {
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

    explicit constexpr FELiftDivergence(const FEFunctionType fe_function)
      : fefunction(std::move(fe_function))
    {
    }

    constexpr FELiftDivergence<FEFunctionType>&
    operator+=(const FELiftDivergence<FEFunctionType>& other_function)
    {
      fefunction += other_function.fefunction;
      return *this;
    };

    const FEFunctionType&
    get_fefunction() const
    {
      return fefunction;
    }

    constexpr auto
    operator-() const
    {
      return FELiftDivergence(-fefunction);
    }

    template <typename Number>
    constexpr
      typename std::enable_if_t<std::is_arithmetic<Number>::value, FELiftDivergence<FEFunctionType>>
      operator*(const Number scalar_factor_) const
    {
      return FELiftDivergence<FEFunctionType>(
        FEFunctionType(fefunction.scalar_factor * scalar_factor_));
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
    using Base::operator+=;
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
    using Base::operator+=;

    explicit constexpr FECurl(const FEFunction<rank - 1, dim, idx>& fefunction)
      : Base(fefunction.scalar_factor)
    {
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
    using Base::operator+=;

    explicit constexpr FEGradient(const FEFunction<rank - 1, dim, idx>& fefunction)
      : Base(fefunction.scalar_factor)
    {
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
    using Base::operator+=;

    explicit FELaplacian(const FEGradient<rank + 1, dim, idx>& fe_function)
      : Base(fe_function.scalar_factor)
    {
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
    using Base::operator+=;
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
    using Base::operator+=;

    explicit FEHessian(const FEGradient<rank - 1, dim, idx>& fefunction)
      : FEHessian(fefunction.scalar_factor)
    {
    }
  };

  /**
   * Utility function to return a FEGradient object
   * given a FEFunction
   *
   */
  template <int rank, int dim, unsigned int idx>
  FEGradient<rank + 1, dim, idx> constexpr grad(const FEFunction<rank, dim, idx>& f)
  {
    return FEGradient<rank + 1, dim, idx>(f);
  }

  /**
   * Utility function to return a FEDivergence object
   * given a FEFunction
   *
   */
  template <int rank, int dim, unsigned int idx>
  FEDivergence<rank - 1, dim, idx> constexpr div(const FEFunction<rank, dim, idx>& f)
  {
    return FEDivergence<rank - 1, dim, idx>(f);
  }

  /**
   * Utility function to return a FEHessian object
   * given a FEGradient
   *
   */
  template <int rank, int dim, unsigned int idx>
  FEHessian<rank + 1, dim, idx> constexpr grad(const FEGradient<rank, dim, idx>& f)
  {
    return FEHessian<rank + 1, dim, idx>(f);
  }

  /**
   * Utility function to return a FELaplacian object
   * given a FEGradient
   *
   */
  template <int rank, int dim, unsigned int idx>
  FELaplacian<rank - 1, dim, idx> constexpr div(const FEGradient<rank, dim, idx>& f)
  {
    return FELaplacian<rank - 1, dim, idx>(f);
  }

  /**
   * Overloading function to scale an FE function with a scalar factor
   *
   */
  template <typename Number, class A>
  constexpr typename std::enable_if_t<Traits::fe_function_set_type<A>::value != ObjectType::none &&
                                        std::is_arithmetic<Number>::value,
                                      A>
  operator*(const Number scalar_factor, const A& a)
  {
    return a * scalar_factor;
  }

  template <class A, class B>
  inline auto
  sum(const A& a, const B& b)
  {
    return a + b;
  }

  template <>
  inline auto
  sum<std::string, std::string>(const std::string& a, const std::string& b)
  {

    if (b[0] == '-')
      return a + b;
    return a + "+" + b;
  }

  template <typename... Types>
  class SumFEFunctions;

  /**
   * Weak formulation of PDE equations gives rise to forms (linear,bilinear
   * etc). These may consist of several FE functions tested with test Function
   * as a sum. It is possible to define sum/difference of such FE Functions using
   * SumFEFunctions class. The summation is achieved by operator overloading.
   * Difference is summation with a negative sign and can be easily achieved.
   *
   * <h3>Implementation</h3>
   * The summation process does not physically add something. Rather, it
   * maintains a static container (also see \ref FEDatas) of the FE Functions
   * An example would be:
   * <code>
                auto sum3 = fe_function1 + fe_function1 + fe_function3;
   * </code>
   * This will be stored as:
   * *   @verbatim
   *		SumFEFunctions<FEFunction>  --> holds fe_function1
   *		     ^
   *		     |
   *		     |
   *		SumFEFunctions<FEFunction,FEFunction> --> holds fe_function2
   *		     ^
   *		     |
   *		     |
   *		SumFEFunctions<FEFunction,FEFunction,FEFunction> --> holds fe_function3
   *   @endverbatim
   */
  template <class FEFunction>
  class SumFEFunctions<FEFunction>
  {
  public:
    using SummandType = FEFunction;
    using TensorTraits =
      Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;
    static constexpr unsigned int count = 0;

    template <class OtherType>
    explicit constexpr SumFEFunctions(const SumFEFunctions<OtherType>& f)
      : summand([&f]() {
        if constexpr(std::is_base_of<FEFunctionBaseBase<OtherType>, OtherType>::value) return f
            .get_summand()
            .scalar_factor;
        else
          return f.get_summand();
      }())
    {
    }

    explicit constexpr SumFEFunctions(const FEFunction summand_)
      : summand(std::move(summand_))
    {
      static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                    "You need to construct this with a FEFunction object!");
    }

    /**
     * Operator overloading to create a SumFEFunctions from
     * two FEFunction objects
     *
     */
    template <class NewFEFunction>
    constexpr auto
    operator+(const NewFEFunction& new_summand) const
    {
      static_assert(Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none,
                    "Only FEFunction objects can be added!");
      static_assert(TensorTraits::dim == NewFEFunction::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(TensorTraits::rank == NewFEFunction::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
      return SumFEFunctions<NewFEFunction, FEFunction>(new_summand, summand);
    }

    /**
     * Operator overloading to create a SumFEFunctions from
     * two FEFunction objects
     *
     */
    template <class NewFEFunction>
    constexpr auto
    operator-(const NewFEFunction& new_summand) const
    {
      return operator+(-new_summand);
    }

    template <class... ParameterTypes>
    auto
    value(const ParameterTypes&... parameters) const
    {
      return summand.value(parameters...);
    }

    /**
     * Wrapper around set_evaluation_flags function of FEFunction
     *
     */
    template <class FEEvaluation>
    static void
    set_evaluation_flags(FEEvaluation& phi)
    {
      FEFunction::set_evaluation_flags(phi);
    }

    /**
     * Returns the FEFunction object held by this object
     *
     */
    constexpr FEFunction
    get_summand() const
    {
      return summand;
    }

    /**
     * Unary minus operator overloading to get -SumFEFunctions
     *
     */
    constexpr SumFEFunctions<FEFunction>
    operator-() const
    {
      // create a copy
      return (*this) * -1.;
    }

    /**
     * Scale all FEFunctions of SumFEFunctions by a scalar factor
     *
     */
    template <typename Number>
    constexpr
      typename std::enable_if<std::is_arithmetic<Number>::value, SumFEFunctions<FEFunction>>::type
      operator*(const Number scalar_factor) const
    {
      return SumFEFunctions<FEFunction>(summand * scalar_factor);
    }

  private:
    const FEFunction summand;
  };

  /**
   * @brief Class to provide Sum of FE Functions.
   * This is for variadic template definition of the SumFEFunctions class.
   * Please refer to the documentation of the previous class
   */
  template <class FEFunction, typename... Types>
  class SumFEFunctions<FEFunction, Types...> : public SumFEFunctions<Types...>
  {
  public:
    using SummandType = FEFunction;
    using TensorTraits =
      Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;
    using Base = SumFEFunctions<Types...>;
    static constexpr unsigned int count = Base::count + 1;

    template <class OtherType, typename... OtherTypes,
              typename std::enable_if<sizeof...(OtherTypes) == sizeof...(Types)>::type* = nullptr>
    explicit constexpr SumFEFunctions(const SumFEFunctions<OtherType, OtherTypes...>& f)
      : SumFEFunctions<Types...>(static_cast<SumFEFunctions<OtherTypes...>>(f))
      , summand([&f]() {
        if constexpr(std::is_base_of<FEFunctionBaseBase<OtherType>, OtherType>::value) return f
            .get_summand()
            .scalar_factor;
        else
          return f.get_summand();
      }())
    {
    }

    /**
     * Actual evaluation of sum of the FE Function values in a Matrix Free
     * context
     *
     */
    template <class... ParameterTypes>
    auto
    value(const ParameterTypes&... parameters) const
    {
      const auto own_value = summand.value(parameters...);
      const auto other_value = Base::value(parameters...);
      assert_is_compatible(own_value, other_value);
      return sum(own_value, other_value);
    }

    /**
     * Wrapper around set_evaluation_flags of FEFunction
     *
     */
    template <class FEEvaluation>
    static void
    set_evaluation_flags(FEEvaluation& phi)
    {
      FEFunction::set_evaluation_flags(phi);
      Base::set_evaluation_flags(phi);
    }

    explicit constexpr SumFEFunctions(const FEFunction summand_, const Types... old_sum)
      : Base(std::move(old_sum...))
      , summand(std::move(summand_))
    {
      static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                    "You need to construct this with a FEFunction object!");
      static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
    }

    constexpr SumFEFunctions(const FEFunction& summand_, const SumFEFunctions<Types...>& old_sum)
      : Base(old_sum)
      , summand(summand_)
    {
      static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                    "You need to construct this with a FEFunction object!");
      static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
    }

    /**
     * Operator overloading to add FEFunction to existing SumFEFunctions
     *
     */
    template <class NewFEFunction,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                Traits::fe_function_set_type<NewFEFunction>::value ==
                  Traits::fe_function_set_type<FEFunction>::value &&
                !Traits::is_fe_function_sum<NewFEFunction>::value>::type* unused = nullptr>
    constexpr auto
    operator+(const NewFEFunction& new_summand) const
    {
      return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_summand, *this);
    }

    /**
     * Operator overloading to add two SumFEFunctions
     *
     */
    template <class NewFEFunction1, class NewFEFunction2, typename... NewTypes,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction1>::value != ObjectType::none &&
                Traits::fe_function_set_type<NewFEFunction1>::value ==
                  Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
    constexpr auto
    operator+(const SumFEFunctions<NewFEFunction1, NewFEFunction2, NewTypes...>& new_sum) const
    {
      return SumFEFunctions<NewFEFunction1, FEFunction, Types...>(new_sum.get_summand(), *this) +
             SumFEFunctions<NewFEFunction2, NewTypes...>(
               static_cast<const SumFEFunctions<NewFEFunction2, NewTypes...>&>(new_sum));
    }

    /**
     * Operator overloading to add two SumFEFunctions
     *
     */
    template <class NewFEFunction,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                Traits::fe_function_set_type<NewFEFunction>::value ==
                  Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
    constexpr auto
    operator+(const SumFEFunctions<NewFEFunction>& new_sum) const
    {
      return SumFEFunctions<NewFEFunction, FEFunction, Types...>(new_sum.get_summand(), *this);
    }

    /**
     * Operator overloading to subtract a FEFunction from a SumFEFunction
     *
     */
    template <class NewFEFunction,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                Traits::fe_function_set_type<NewFEFunction>::value ==
                  Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
    constexpr auto
    operator-(const NewFEFunction& new_summand) const
    {
      return operator+(-new_summand);
    }

    /**
     * Unary minus operator overloading to get -SumFEFunction
     *
     */
    constexpr auto
    operator-() const
    {
      return (*this) * -1.;
    }

    /**
     * Multiply all FEFunction objects of this SumFEFunction with a scalar factor
     *
     */
    template <typename Number>
    constexpr typename std::enable_if<std::is_arithmetic<Number>::value,
                                      SumFEFunctions<FEFunction, Types...>>::type
    operator*(const Number scalar_factor) const
    {
      return SumFEFunctions<FEFunction, Types...>(
        summand * scalar_factor, static_cast<SumFEFunctions<Types...>>(*this) * scalar_factor);
    }

    /**
     * Operator overloading to subtract a SumFEFuction from a SumFEFunction
     *
     */
    template <class NewFEFunction, typename... NewTypes,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                Traits::fe_function_set_type<NewFEFunction>::value ==
                  Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
    constexpr auto
    operator-(const SumFEFunctions<NewFEFunction, NewTypes...>& new_sum) const
    {
      return operator+(-new_sum);
    }

    /**
     * Operator overloading to subtract a SumFEFuction from a SumFEFunction
     *
     */
    template <class NewFEFunction,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                Traits::fe_function_set_type<NewFEFunction>::value ==
                  Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
    constexpr auto
    operator-(const SumFEFunctions<NewFEFunction>& new_sum) const
    {
      return operator+(-new_sum);
    }

    constexpr FEFunction
    get_summand() const
    {
      return summand;
    }

  private:
    const FEFunction summand;
  };

  /**
   * Operator overloading to add two FEFunction objects to form a SumFEFuction object
   *
   */
  template <class FEFunction1, class FEFunction2,
            typename std::enable_if<
              Traits::fe_function_set_type<FEFunction1>::value != ObjectType::none &&
              Traits::fe_function_set_type<FEFunction1>::value ==
                Traits::fe_function_set_type<FEFunction2>::value &&
              !Traits::is_fe_function_sum<FEFunction1>::value &&
              !Traits::is_fe_function_sum<FEFunction2>::value>::type* unused = nullptr>
  constexpr auto
  operator+(const FEFunction1& old_fe_function, const FEFunction2& new_fe_function)
  {

    static_assert(FEFunction1::TensorTraits::dim == FEFunction2::TensorTraits::dim,
                  "You can only add tensors of equal dimension!");
    static_assert(FEFunction1::TensorTraits::rank == FEFunction2::TensorTraits::rank,
                  "You can only add tensors of equal rank!");
    return SumFEFunctions<FEFunction2, FEFunction1>(new_fe_function, old_fe_function);
  }

  /**
   * Operator overloading to add a FEFunction object with a SumFEFuctions object
   *
   */
  template <
    class FEFunction, typename... Types,
    typename std::enable_if<Traits::fe_function_set_type<FEFunction>::value != ObjectType::none &&
                            !Traits::is_fe_function_sum<FEFunction>::value>::type* unused = nullptr>
  constexpr auto
  operator+(const FEFunction& new_fe_function, const SumFEFunctions<Types...>& old_fe_function)
  {
    return old_fe_function + new_fe_function;
  }

  /**
   * Operator overloading to subtract two FEFunction objects to form a SumFEFuction object
   *
   */
  template <class FEFunction1, class FEFunction2,
            typename std::enable_if<
              Traits::fe_function_set_type<FEFunction1>::value != ObjectType::none &&
              Traits::fe_function_set_type<FEFunction1>::value ==
                Traits::fe_function_set_type<FEFunction2>::value &&
              !Traits::is_fe_function_sum<FEFunction1>::value &&
              !Traits::is_fe_function_sum<FEFunction2>::value>::type* unused = nullptr>
  constexpr auto
  operator-(const FEFunction1& old_fe_function, const FEFunction2& new_fe_function)
  {
    return old_fe_function + (-new_fe_function);
  }

  /**
   * Operator overloading to subtract an FEFunction object from a SumFEFuction object
   *
   */
  template <
    class FEFunction, typename... Types,
    typename std::enable_if<Traits::fe_function_set_type<FEFunction>::value != ObjectType::none &&
                            !Traits::is_fe_function_sum<FEFunction>::value>::type* unused = nullptr>
  constexpr auto
  operator-(const FEFunction& new_fe_function, const SumFEFunctions<Types...>& old_fe_function)
  {
    return -(old_fe_function - new_fe_function);
  }

  template <class... Types>
  constexpr auto
  optimize(const SumFEFunctions<Types...>& sum)
  {
    // construct a list of processed forms and add similar ones.
    auto list = internal::optimize::create_list(sum, std::tuple<>{});

    // then create a new object from this list and return it.
    return internal::optimize::create_types<0>(std::move(list));
  }

  template <class... Types>
  constexpr auto
  optimize(const ProductFEFunctions<Types...>& product)
  {
    return product;
  }

  template <class A, class B>
  inline auto
  product(const A& a, const B& b)
  {
    return a * b;
  }

  template <>
  inline auto
  product<std::string, std::string>(const std::string& a, const std::string& b)
  {
    return a + R"( \cdot )" + b;
  }

  /**
   * See \ref SumFEFunctions.
   * It is possible to define product of such FE Functions using
   * ProductFEFunctions class. The product is achieved by operator overloading.
   *
   * <h3>Implementation</h3>
   * The product process does not physically multiply something. Rather, it
   * maintains a static container (also see \ref FEDatas) of the FE Functions
   * An example would be:
   * <code>
                auto prod1 = fe_function1 * fe_function1 * fe_function3;
   * </code>
   * This will be stored as:
   * *   @verbatim
   *		ProductFEFunctions<FEFunction>  --> holds fe_function1
   *		     ^
   *		     |
   *		     |
   *		ProductFEFunctions<FEFunction,FEFunction> --> holds fe_function2
   *		     ^
   *		     |
   *		     |
   *		ProductFEFunctions<FEFunction,FEFunction,FEFunction> --> holds fe_function3
   *   @endverbatim
   */
  template <class FEFunction>
  class ProductFEFunctions<FEFunction>
  {
  public:
    using TensorTraits =
      Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;
    static constexpr unsigned int n = 1;

    explicit constexpr ProductFEFunctions(FEFunction factor_)
      : factor(std::move(factor_))
    {
      static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                    "You need to construct this with a FEFunction object!");
    }

    template <class OtherType>
    explicit constexpr ProductFEFunctions(const ProductFEFunctions<OtherType>& f)
      : factor(f.get_factor())
    {
    }

    // prodfefunc * fefunc
    /**
     * Operator overloading to multiply a ProductFEFunctions with an FEFunction
     *
     */
    template <class NewFEFunction,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction>::value ==
                    Traits::fe_function_set_type<FEFunction>::value,
                ProductFEFunctions<NewFEFunction, FEFunction>>::type* unused = nullptr>
    constexpr ProductFEFunctions<NewFEFunction, FEFunction> operator*(
      const NewFEFunction& new_factor) const
    {
      static_assert(Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none,
                    "Only FEFunction objects can be added!");
      static_assert(TensorTraits::dim == NewFEFunction::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(TensorTraits::rank == NewFEFunction::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
      return ProductFEFunctions<NewFEFunction, FEFunction>(new_factor, factor);
    }

    /**
     * Wrapper around value function of FEFunction
     *
     */
    template <class... ParameterTypes>
    auto
    value(const ParameterTypes&... parameters) const
    {
      return factor.value(parameters...);
    }

    /**
     * Wrapper around set_evaluation_flags function of FEFunction
     *
     */
    template <class FEEvaluation>
    static void
    set_evaluation_flags(FEEvaluation& phi)
    {
      FEFunction::set_evaluation_flags(phi);
    }

    /**
     * Returns the FEFunction object held by this object
     *
     */
    constexpr const FEFunction&
    get_factor() const
    {
      return factor;
    }

    // prodfefunc * number
    /**
     * Multiply all components of ProductFEFunctions with a scalar factor
     *
     */
    template <typename Number,
              typename std::enable_if<std::is_arithmetic<Number>::value>::type* = nullptr> constexpr auto operator*(const Number scalar_factor) const
    {
      return ProductFEFunctions<FEFunction>(factor * scalar_factor);
    }

    /**
     * Unary minus operator overloading to get -ProductFEFunctions
     *
     */
    constexpr ProductFEFunctions<FEFunction>
    operator-() const
    {
      return (*this) * -1.;
    }

  private:
    FEFunction factor;
  };

  /**
   * @brief Class to provide Product of FE Functions.
   * This is for variadic template definition of the ProductFEFunctions class.
   * Please refer to the documentation of the previous class
   */
  template <class FEFunction, typename... Types>
  class ProductFEFunctions<FEFunction, Types...> : public ProductFEFunctions<Types...>
  {
  public:
    using TensorTraits =
      Traits::Tensor<FEFunction::TensorTraits::rank, FEFunction::TensorTraits::dim>;
    using Base = ProductFEFunctions<Types...>;
    static constexpr unsigned int n = Base::n + 1;

    template <class OtherType, typename... OtherTypes,
              typename std::enable_if<sizeof...(OtherTypes) == sizeof...(Types)>::type* = nullptr>
    explicit constexpr ProductFEFunctions(const ProductFEFunctions<OtherType, OtherTypes...>& f)
      : ProductFEFunctions<Types...>(static_cast<ProductFEFunctions<OtherTypes...>>(f))
      , factor(f.get_factor())
    {
    }

    /**
     * Actual evaluation of sum of the FE Function values in a Matrix Free
     * context
     *
     */
    template <class... ParameterTypes>
    auto
    value(const ParameterTypes&... parameters) const
    {
      const auto own_value = factor.value(parameters...);
      const auto other_value = Base::value(parameters...);
      assert_is_compatible(own_value, other_value);
      return product(own_value, other_value);
    }

    /**
     * Wrapper around set_evaluation_flags of FEFunction
     *
     */
    template <class FEEvaluation>
    static void
    set_evaluation_flags(FEEvaluation& phi)
    {
      FEFunction::set_evaluation_flags(phi);
      Base::set_evaluation_flags(phi);
    }

    constexpr ProductFEFunctions(const FEFunction factor_, const Types... old_product)
      : Base(std::move(old_product...))
      , factor(std::move(factor_))
    {
      static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                    "You need to construct this with a FEFunction object!");
      static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
    }

    constexpr ProductFEFunctions(const FEFunction factor_,
                                 const ProductFEFunctions<Types...> old_product)
      : Base(std::move(old_product))
      , factor(std::move(factor_))
    {
      static_assert(Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                    "You need to construct this with a FEFunction object!");
      static_assert(TensorTraits::dim == Base::TensorTraits::dim,
                    "You can only add tensors of equal dimension!");
      static_assert(TensorTraits::rank == Base::TensorTraits::rank,
                    "You can only add tensors of equal rank!");
    }

    // prodfefunc * fefunc
    /**
     * Operator overloading to multiply FEFunction to existing ProductFEFunctions
     *
     */
    template <class NewFEFunction,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                  Traits::fe_function_set_type<NewFEFunction>::value ==
                    Traits::fe_function_set_type<FEFunction>::value,
                ProductFEFunctions<NewFEFunction, FEFunction, Types...>>::type* unused = nullptr>
    constexpr auto operator*(const NewFEFunction& new_factor) const
    {
      return ProductFEFunctions<NewFEFunction, FEFunction, Types...>(new_factor, *this);
    }

    /**
     * Unary minus operator overloading to negate a ProductFEFunctions
     *
     */
    constexpr ProductFEFunctions<FEFunction, Types...>
    operator-() const
    {
      return (*this) * -1.;
    }

    // prodfefunc * number
    /**
     * Multiply all components of ProductFEFunctions with a scalar factor
     *
     */
    template <typename Number,
              typename std::enable_if<std::is_arithmetic<Number>::value>::type* = nullptr> constexpr auto operator*(const Number scalar_factor) const
    {
      return ProductFEFunctions<FEFunction, Types...>(
        factor * scalar_factor, static_cast<ProductFEFunctions<Types...>>(*this));
    }

    // prodfefunc * prodfefunc
    /**
     * Operator overloading to multiply two ProductFEFunctions
     *
     */
    template <class NewFEFunction1, class NewFEFunction2, typename... NewTypes,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction1>::value != ObjectType::none &&
                Traits::fe_function_set_type<NewFEFunction1>::value ==
                  Traits::fe_function_set_type<FEFunction>::value &&
                Traits::fe_function_set_type<NewFEFunction2>::value != ObjectType::none &&
                Traits::fe_function_set_type<NewFEFunction2>::value ==
                  Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
    constexpr auto operator*(
      const ProductFEFunctions<NewFEFunction1, NewFEFunction2, NewTypes...>& new_product) const
    {
      return ProductFEFunctions<NewFEFunction1, FEFunction, Types...>(new_product.get_factor(),
                                                                      *this) *
             ProductFEFunctions<NewFEFunction2, NewTypes...>(
               static_cast<const ProductFEFunctions<NewFEFunction2, NewTypes...>&>(new_product));
    }

    // prodfefunc * prodfefunc
    /**
     * Operator overloading to multiply two ProductFEFunctions
     *
     */
    template <class NewFEFunction,
              typename std::enable_if<
                Traits::fe_function_set_type<NewFEFunction>::value != ObjectType::none &&
                Traits::fe_function_set_type<NewFEFunction>::value ==
                  Traits::fe_function_set_type<FEFunction>::value>::type* unused = nullptr>
    constexpr auto operator*(const ProductFEFunctions<NewFEFunction>& new_product) const
    {
      return ProductFEFunctions<NewFEFunction, FEFunction, Types...>(new_product.get_factor(),
                                                                     *this);
    }

    constexpr const FEFunction&
    get_factor() const
    {
      return factor;
    }

  private:
    FEFunction factor;
  };

  // fefunc * fefunc
  /**
   * Operator overloading to multiply two FEFunction objects to get a
   * ProductFEFunctions
   *
   */
  template <
    class FEFunction1, class FEFunction2,
    typename std::enable_if<Traits::fe_function_set_type<FEFunction1>::value != ObjectType::none &&
                              Traits::fe_function_set_type<FEFunction1>::value ==
                                Traits::fe_function_set_type<FEFunction2>::value &&
                              !Traits::is_fe_function_product<FEFunction1>::value &&
                              !Traits::is_fe_function_product<FEFunction2>::value,
                            ProductFEFunctions<FEFunction2, FEFunction1>>::type* unused = nullptr>
  constexpr auto operator*(const FEFunction1& old_fe_function, const FEFunction2& new_fe_function)
  {
    static_assert(FEFunction1::TensorTraits::dim == FEFunction2::TensorTraits::dim,
                  "You can only multiply tensors of equal dimension!");
    static_assert(FEFunction1::TensorTraits::rank == FEFunction2::TensorTraits::rank,
                  "You can only multiply tensors of equal rank!");
    return ProductFEFunctions<FEFunction2, FEFunction1>(new_fe_function, old_fe_function);
  }

  // Note: this function changes the order of template parameters compared with its other sibling
  // functions which perform multiplication operation on prodfefunc and fefunc
  // fefunc * prodfefunc
  /**
   * Operator overloading to multiply an FEFunction object to a
   * ProductFEFunctions
   *
   */
  template <
    class FEFunction, typename... Types,
    typename std::enable_if<Traits::fe_function_set_type<FEFunction>::value != ObjectType::none,
                            ProductFEFunctions<FEFunction, Types...>>::type* unused = nullptr>
  constexpr auto operator*(const FEFunction& new_fe_function,
                           const ProductFEFunctions<Types...>& old_fe_function)
  {
    return old_fe_function * new_fe_function;
  }
}
} // namespace CFL

#endif
