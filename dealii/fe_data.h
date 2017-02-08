#ifndef FE_DATA_H
#define FE_DATA_H

#include <cfl/traits.h>
#include <cfl/dealii_matrixfree.h>

template <int fe_degree, int n_components, int dim, unsigned int fe_no, unsigned int max_fe_degree,
          typename Number = double>
class FEData
{
public:
  typedef typename dealii::FEEvaluation<dim, fe_degree, max_fe_degree + 1, n_components, Number>
  FEEvaluationType;
  typedef Number NumberType;
  typedef CFL::Traits::Tensor<(n_components > 1 ? 1 : 0), dim> TensorTraits;
  static constexpr unsigned int fe_number = fe_no;
  static constexpr unsigned int max_degree = max_fe_degree;

  explicit FEData(const FiniteElement<dim> &fe)
  {
    (void)fe;
    static_assert(fe_degree <= max_degree, "fe_degree must not be greater than max_degree!");
    Assert(fe.degree == fe_degree, ExcIndexRange(fe.degree, fe_degree, fe_degree));
    AssertDimension(fe.n_components(), n_components);
  }
};

namespace CFL
{
  namespace Traits
  {
    template <int fe_degree, int n_components, int dim, unsigned int fe_no, unsigned int max_degree,
              typename Number>
    struct is_fe_data<FEData<fe_degree, n_components, dim, fe_no, max_degree, Number>>
    {
      static const bool value = true;
    };
  }
}

template <typename... Types>
class FEDatas
{
public:
  FEDatas() = delete;

  template <class FEData>
  FEDatas<FEData, Types...> operator, (const FEData &)
  {
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "You need to construct this with a FEData object!");
    return FEDatas<FEData, Types...>();
  }
};

template <class FEData>
class FEDatas<FEData>
{
public:
  typedef typename FEData::FEEvaluationType FEEvaluationType;
  typedef typename FEData::TensorTraits TensorTraits;
  typedef typename FEData::NumberType NumberType;
  static constexpr unsigned int fe_number = FEData::fe_number;
  static constexpr unsigned int max_degree = FEData::max_degree;

  FEDatas()
    : integrate_values(false)
    , integrate_gradients(false)
    , evaluate_values(false)
    , evaluate_gradients(false)
    , evaluate_hessians(false)
    , initialized(false)
  {
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "You need to construct this with a FEData object!");
  }

  FEDatas(const FEData &)
    : integrate_values(false)
    , integrate_gradients(false)
    , evaluate_values(false)
    , evaluate_gradients(false)
    , evaluate_hessians(false)
    , initialized(false)
  {
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "You need to construct this with a FEData object!");
  }

  FEDatas(const FEDatas<FEData> &fe_datas)
    : integrate_values(false)
    , integrate_gradients(false)
    , evaluate_values(false)
    , evaluate_gradients(false)
    , evaluate_hessians(false)
    , initialized(false)
  {
    (void)fe_datas;
    Assert(!fe_datas.initialized, ExcNotImplemented());
  }

  template <unsigned int fe_number_extern>
  static constexpr bool
  rank()
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return TensorTraits::rank;
  }

  template <class NewFEData>
  FEDatas<NewFEData, FEData> operator, (const NewFEData &)
  {
    static_assert(CFL::Traits::is_fe_data<NewFEData>::value, "Only FEData objects can be added!");

    return FEDatas<NewFEData, FEData>();
  }

  template <typename Cell>
  void
  reinit(const Cell &cell)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Reinit FEDatas " << fe_number << std::endl;
#endif
    Assert(fe_evaluation.get() != nullptr, ExcInternalError());
    fe_evaluation->reinit(cell);
  }

  template <typename VectorType>
  void
  read_dof_values(const VectorType &vector)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Read DoF values " << fe_number << std::endl;
#endif
    Assert(fe_evaluation.get() != nullptr, ExcInternalError());
    if
    constexpr(CFL::Traits::is_block_vector<VectorType>::value)
      fe_evaluation->read_dof_values(vector.block(fe_number));
    else
      fe_evaluation->read_dof_values(vector);
  }

  template <unsigned int fe_number_extern>
  void
  set_integration_flags(bool integrate_value, bool integrate_gradient)
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    integrate_values |= integrate_value;
    integrate_gradients |= integrate_gradient;
#ifdef DEBUG_OUTPUT
    std::cout << "integrate value end: " << fe_number << " " << integrate_values << " "
              << integrate_value << std::endl;
    std::cout << "integrate gradients end: " << fe_number << " " << integrate_gradients << " "
              << integrate_gradient << std::endl;
#endif
  }

  template <unsigned int fe_number_extern>
  void
  set_evaluation_flags(bool evaluate_value, bool evaluate_gradient, bool evaluate_hessian)
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    evaluate_values |= evaluate_value;
    evaluate_gradients |= evaluate_gradient;
    evaluate_hessians |= evaluate_hessian;
  }

  template <typename VectorType>
  void
  distribute_local_to_global(VectorType &vector)
  {
    if (integrate_values | integrate_gradients)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Distribute DoF values " << fe_number << std::endl;
#endif
        Assert(fe_evaluation.get() != nullptr, ExcInternalError());
        if
        constexpr(CFL::Traits::is_block_vector<VectorType>::value)
          fe_evaluation->distribute_local_to_global(vector.block(fe_number));
        else
#ifdef DEBUG_OUTPUT
#endif
          fe_evaluation->distribute_local_to_global(vector);
      }
  }

  template <int dim, typename OtherNumber>
  void
  initialize(const MatrixFree<dim, OtherNumber> &mf)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Initialize FEDatas " << fe_number << std::endl;
#endif
    static_assert(std::is_same<NumberType, OtherNumber>::value,
                  "Number type of MatrixFree and FEDatas has to match!");
    fe_evaluation.reset(new typename FEData::FEEvaluationType(mf, fe_number));
    initialized = true;
  }

  void
  evaluate()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Evaluate FEDatas " << fe_number << " " << evaluate_values << " "
              << evaluate_gradients << " " << evaluate_hessians << std::endl;
#endif
    Assert(fe_evaluation.get() != nullptr, ExcInternalError());
    fe_evaluation->evaluate(evaluate_values, evaluate_gradients, evaluate_hessians);
  }

  template <unsigned int fe_number_extern = fe_number>
  static constexpr unsigned int
  get_n_q_points()
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return FEData::FEEvaluationType::n_q_points;
  }

  template <unsigned int fe_number_extern>
  auto get_gradient(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_evaluation->get_gradient(q);
  }

  template <unsigned int fe_number_extern>
  auto get_symmetric_gradient(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get symmetric gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_evaluation->get_symmetric_gradient(q);
  }

  template <unsigned int fe_number_extern>
  auto get_divergence(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get divergence FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_evaluation->get_divergence(q);
  }

  template <unsigned int fe_number_extern>
  auto get_laplacian(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get laplacian FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_evaluation->get_laplacian(q);
  }

  template <unsigned int fe_number_extern>
  auto get_hessian_diagonal(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get hessian_diagonal FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_evaluation->get_hessian_diagonal(q);
  }

  template <unsigned int fe_number_extern>
  auto get_hessian(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get hessian FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_evaluation->get_hessian(q);
  }

  template <unsigned int fe_number_extern>
  auto get_value(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get value FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_evaluation->get_value(q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_gradient(const ValueType &value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit gradient FEDatas " << fe_number << " " << q << std::endl;
//    for (unsigned int i=0; i<2; ++i)
//      std::cout << " " << value[i][0];
//    std::cout << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_evaluation->submit_gradient(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_curl(const ValueType &value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit curl FEDatas " << fe_number << " " << q << std::endl;
    for (unsigned int i = 0; i < 2; ++i)
      std::cout << " " << value[i][0];
    std::cout << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_evaluation->submit_curl(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_divergence(const ValueType &value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit divergence FEDatas " << fe_number << " " << q << std::endl;
    for (unsigned int i = 0; i < 2; ++i)
      std::cout << " " << value[i][0];
    std::cout << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_evaluation->submit_divergence(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_symmetric_gradient(const ValueType &value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit symmetric gradient FEDatas " << fe_number << " " << q << std::endl;
    for (unsigned int i = 0; i < 2; ++i)
      std::cout << " " << value[i][0];
    std::cout << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_evaluation->submit_symmetric_gradient(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_value(const ValueType &value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit value FEDatas" << fe_number << " " << q << std::endl;
//    for (unsigned int i=0; i<2; ++i)
//      std::cout << " " << value[i][0];
//    std::cout << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_evaluation->submit_value(value, q);
  }

  void
  integrate()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "integrate FEDatas " << fe_number << " " << integrate_values << " "
              << integrate_gradients << std::endl;
#endif
    if (integrate_values | integrate_gradients)
      fe_evaluation->integrate(integrate_values, integrate_gradients);
  }

  const typename FEData::FEEvaluationType &
  get_fe_evaluation()
  {
    return fe_evaluation();
  }

  template <unsigned int fe_number_extern>
  unsigned int
  dofs_per_cell()
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_evaluation->dofs_per_cell;
  }

  template <unsigned int fe_number_extern>
  static constexpr unsigned int
  tensor_dofs_per_cell()
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return FEData::FEEvaluationType::tensor_dofs_per_cell;
  }

  template <unsigned int fe_number_extern>
  auto begin_dof_values()
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_evaluation->begin_dof_values();
  }

private:
  std::unique_ptr<typename FEData::FEEvaluationType> fe_evaluation = nullptr;
  bool integrate_values = false;
  bool integrate_gradients = false;
  bool evaluate_values = false;
  bool evaluate_gradients = false;
  bool evaluate_hessians = false;
  bool initialized = false;
};

template <class FEData, typename... Types>
class FEDatas<FEData, Types...> : public FEDatas<Types...>
{
public:
  typedef typename FEData::FEEvaluationType FEEvaluationType;
  typedef typename FEData::TensorTraits TensorTraits;
  typedef typename FEData::NumberType NumberType;
  static constexpr unsigned int fe_number = FEData::fe_number;
  static constexpr unsigned int max_degree = FEDatas<Types...>::max_degree;

  template <unsigned int fe_number_extern>
  static constexpr unsigned int
  rank()
  {
    if
    constexpr(fe_number == fe_number_extern) return TensorTraits::rank;
    else
      return FEDatas<Types...>::template rank<fe_number_extern>();
  }

  template <int dim, typename OtherNumber>
  auto initialize(const MatrixFree<dim, OtherNumber> &mf)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Initialize FEDatas " << fe_number << std::endl;
    std::cout << "integrate_valus at initialize: " << integrate_values << std::endl;
#endif
    static_assert(std::is_same<NumberType, OtherNumber>::value,
                  "Number type of MatrixFree and FEDatas do not match!");
    fe_evaluation.reset(new typename FEData::FEEvaluationType(mf, fe_number));
    FEDatas<Types...>::initialize(mf);
    initialized = true;
  }

  template <typename Cell>
  void
  reinit(const Cell &cell)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Reinit FEDatas " << fe_number << std::endl;
#endif
    Assert(fe_evaluation.get() != nullptr, ExcInternalError());
    fe_evaluation->reinit(cell);
    FEDatas<Types...>::reinit(cell);
  }

  template <typename VectorType>
  void
  read_dof_values(const VectorType &vector)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Read DoF values " << fe_number << std::endl;
#endif
    Assert(fe_evaluation.get() != nullptr, ExcInternalError());
    if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
    {
      fe_evaluation->read_dof_values(vector.block(fe_number));
      FEDatas<Types...>::read_dof_values(vector);
    }
    else
    {
      AssertThrow(false, ExcInternalError());
/*      static_assert(CFL::Traits::is_block_vector<VectorType>::value,
                    "It only makes sense to have multiple FEData objects if "
                    "you provide a block vector.");*/
    }
  }

  template <typename VectorType>
  typename std::enable_if<CFL::Traits::is_block_vector<VectorType>::value, void>::type
  distribute_local_to_global(VectorType &vector)
  {
    if (integrate_values | integrate_gradients)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Distribute DoF values " << fe_number << std::endl;
#endif
        Assert(fe_evaluation.get() != nullptr, ExcInternalError());
        fe_evaluation->distribute_local_to_global(vector.block(fe_number));
      }

    FEDatas<Types...>::distribute_local_to_global(vector);
  }

  template <typename VectorType>
  typename std::enable_if<!CFL::Traits::is_block_vector<VectorType>::value, void>::type
  distribute_local_to_global(const VectorType &)
  {
    static_assert(!CFL::Traits::is_block_vector<VectorType>::value,
                  "It only makes sense to have multiple FEData objects if "
                  "you provide a block vector.");
  }

  void
  evaluate()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Evaluate FEDatas " << fe_number << " " << evaluate_values << " "
              << evaluate_gradients << " " << evaluate_hessians << std::endl;
#endif
    Assert(fe_evaluation.get() != nullptr, ExcInternalError());
    fe_evaluation->evaluate(evaluate_values, evaluate_gradients, evaluate_hessians);
    FEDatas<Types...>::evaluate();
  }

  void
  integrate()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "integrate FEDatas " << fe_number << " " << integrate_values << " "
              << integrate_gradients << std::endl;
#endif
    if (integrate_values | integrate_gradients)
      fe_evaluation->integrate(integrate_values, integrate_gradients);
    FEDatas<Types...>::integrate();
  }

  template <unsigned int fe_number_extern>
  typename std::enable_if<fe_number == fe_number_extern>::type
  set_integration_flags(bool integrate_value, bool integrate_gradient)
  {
    integrate_values |= integrate_value;
    integrate_gradients |= integrate_gradient;
#ifdef DEBUG_OUTPUT
    std::cout << "integrate value: " << fe_number << " " << integrate_values << " "
              << integrate_value << std::endl;
    std::cout << "integrate gradients: " << fe_number << " " << integrate_gradients << " "
              << integrate_gradient << std::endl;
#endif
  }

  template <unsigned int fe_number_extern>
  typename std::enable_if<fe_number != fe_number_extern>::type
  set_integration_flags(bool integrate_value, bool integrate_gradient)
  {
    FEDatas<Types...>::template set_integration_flags<fe_number_extern>(integrate_value,
        integrate_gradient);
  }

  template <unsigned int fe_number_extern>
  typename std::enable_if<fe_number == fe_number_extern>::type
  set_evaluation_flags(bool evaluate_value, bool evaluate_gradient, bool evaluate_hessian)
  {
    evaluate_values |= evaluate_value;
    evaluate_gradients |= evaluate_gradient;
    evaluate_hessians |= evaluate_hessian;
  }

  template <unsigned int fe_number_extern>
  typename std::enable_if<fe_number != fe_number_extern>::type
  set_evaluation_flags(bool evaluate_value, bool evaluate_gradient, bool evaluate_hessian)
  {
    FEDatas<Types...>::template set_evaluation_flags<fe_number_extern>(
      evaluate_value, evaluate_gradient, evaluate_hessian);
  }

  template <unsigned int fe_number_extern = fe_number>
  static constexpr typename std::enable_if<fe_number == fe_number_extern, unsigned int>::type
  get_n_q_points()
  {
    return FEData::FEEvaluationType::n_q_points;
  }

  template <unsigned int fe_number_extern = fe_number>
  static constexpr typename std::enable_if<fe_number != fe_number_extern, unsigned int>::type
  get_n_q_points()
  {
    return FEDatas<Types...>::template get_n_q_points<fe_number_extern>();
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number == fe_number_extern, void>::type>
  auto get_gradient(
    unsigned int q,
    typename std::enable_if<fe_number == fe_number_extern, void>::type * = nullptr) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
    return fe_evaluation->get_gradient(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number != fe_number_extern, void>::type>
  auto get_gradient(
    unsigned int q,
    typename std::enable_if<fe_number != fe_number_extern, void>::type * = nullptr) const
  {
    return FEDatas<Types...>::template get_gradient<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number == fe_number_extern, void>::type>
  auto get_symmetric_gradient(
    unsigned int q,
    typename std::enable_if<fe_number == fe_number_extern, void>::type * = nullptr) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get symmetric gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
    return fe_evaluation->get_symmetric_gradient(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number != fe_number_extern, void>::type>
  auto get_symmetric_gradient(
    unsigned int q,
    typename std::enable_if<fe_number != fe_number_extern, void>::type * = nullptr) const
  {
    return FEDatas<Types...>::template get_symmetric_gradient<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number == fe_number_extern, void>::type>
  auto get_divergence(
    unsigned int q,
    typename std::enable_if<fe_number == fe_number_extern, void>::type * = nullptr) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get divergence FEDatas " << fe_number << " " << q << std::endl;
#endif
    return fe_evaluation->get_divergence(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number != fe_number_extern, void>::type>
  auto get_divergence(
    unsigned int q,
    typename std::enable_if<fe_number != fe_number_extern, void>::type * = nullptr) const
  {
    return FEDatas<Types...>::template get_divergence<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number == fe_number_extern, void>::type>
  auto get_laplacian(
    unsigned int q,
    typename std::enable_if<fe_number == fe_number_extern, void>::type * = nullptr) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get laplacian FEDatas " << fe_number << " " << q << std::endl;
#endif
    return fe_evaluation->get_laplacian(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number != fe_number_extern, void>::type>
  auto get_laplacian(
    unsigned int q,
    typename std::enable_if<fe_number != fe_number_extern, void>::type * = nullptr) const
  {
    return FEDatas<Types...>::template get_laplacian<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number == fe_number_extern, void>::type>
  auto get_hessian_diagonal(
    unsigned int q,
    typename std::enable_if<fe_number == fe_number_extern, void>::type * = nullptr) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get hessian_diagonal FEDatas " << fe_number << " " << q << std::endl;
#endif
    return fe_evaluation->get_hessian_diagonal(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number != fe_number_extern, void>::type>
  auto get_hessian_diagonal(
    unsigned int q,
    typename std::enable_if<fe_number != fe_number_extern, void>::type * = nullptr) const
  {
    return FEDatas<Types...>::template get_hessian_diagonal<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number == fe_number_extern, void>::type>
  auto get_hessian(
    unsigned int q,
    typename std::enable_if<fe_number == fe_number_extern, void>::type * = nullptr) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get hessian FEDatas " << fe_number << " " << q << std::endl;
#endif
    return fe_evaluation->get_hessian(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number != fe_number_extern, void>::type>
  auto get_hessian(
    unsigned int q,
    typename std::enable_if<fe_number != fe_number_extern, void>::type * = nullptr) const
  {
    return FEDatas<Types...>::template get_hessian<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number == fe_number_extern, void>::type>
  auto get_value(
    unsigned int q,
    typename std::enable_if<fe_number == fe_number_extern, void>::type * = nullptr) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get value FEDatas " << fe_number << " " << q << std::endl;
#endif
    return fe_evaluation->get_value(q);
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number != fe_number_extern, void>::type>
  auto get_value(
    unsigned int q,
    typename std::enable_if<fe_number != fe_number_extern, void>::type * = nullptr) const
  {
    return FEDatas<Types...>::template get_value<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number == fe_number_extern, void>::type
  submit_curl(const ValueType &value, unsigned int q)
  {
    fe_evaluation->submit_curl(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number != fe_number_extern, void>::type
  submit_curl(const ValueType &value, unsigned int q)
  {
    FEDatas<Types...>::template submit_curl<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number == fe_number_extern, void>::type
  submit_divergence(const ValueType &value, unsigned int q)
  {
    fe_evaluation->submit_divergence(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number != fe_number_extern, void>::type
  submit_divergence(const ValueType &value, unsigned int q)
  {
    FEDatas<Types...>::template submit_divergence<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number == fe_number_extern, void>::type
  submit_symmetric_gradient(const ValueType &value, unsigned int q)
  {
    fe_evaluation->submit_symmetric_gradient(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number != fe_number_extern, void>::type
  submit_symmetric_gradient(const ValueType &value, unsigned int q)
  {
    FEDatas<Types...>::template submit_symmetric_gradient<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number == fe_number_extern, void>::type
  submit_gradient(const ValueType &value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit gradient FEDatas " << fe_number << " " << q << std::endl;
//    for (unsigned int i=0; i<2; ++i)
//      std::cout << " " << value[i][0];
//    std::cout << std::endl;
#endif
    fe_evaluation->submit_gradient(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number != fe_number_extern, void>::type
  submit_gradient(const ValueType &value, unsigned int q)
  {
    FEDatas<Types...>::template submit_gradient<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number == fe_number_extern, void>::type
  submit_value(const ValueType &value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get value FEDatas " << fe_number << " " << q << std::endl;
//    for (unsigned int i=0; i<value.size(); ++i)
//      std::cout << "value: " << value[i][0] << std::endl;
#endif
    fe_evaluation->submit_value(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  typename std::enable_if<fe_number != fe_number_extern, void>::type
  submit_value(const ValueType &value, unsigned int q)
  {
    FEDatas<Types...>::template submit_value<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern>
  typename std::enable_if<fe_number == fe_number_extern, unsigned int>::type
  dofs_per_cell()
  {
    return fe_evaluation->dofs_per_cell;
  }

  template <unsigned int fe_number_extern>
  typename std::enable_if<fe_number != fe_number_extern, unsigned int>::type
  dofs_per_cell()
  {
    return FEDatas<Types...>::template dofs_per_cell<fe_number_extern>();
  }

  template <unsigned int fe_number_extern>
  static constexpr typename std::enable_if<fe_number == fe_number_extern, unsigned int>::type
  tensor_dofs_per_cell()
  {
    return FEData::FEEvaluationType::tensor_dofs_per_cell;
  }

  template <unsigned int fe_number_extern>
  static constexpr typename std::enable_if<fe_number != fe_number_extern, unsigned int>::type
  tensor_dofs_per_cell()
  {
    return FEDatas<Types...>::template tensor_dofs_per_cell<fe_number_extern>();
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number == fe_number_extern, void>::type>
  auto begin_dof_values(
    typename std::enable_if<fe_number == fe_number_extern, void>::type * = nullptr)
  {
    return fe_evaluation->begin_dof_values();
  }

  template <unsigned int fe_number_extern,
            typename = typename std::enable_if<fe_number != fe_number_extern, void>::type>
  auto begin_dof_values(
    typename std::enable_if<fe_number != fe_number_extern, void>::type * = nullptr)
  {
    return FEDatas<Types...>::template begin_dof_values<fe_number_extern>();
  }

  FEDatas()
    : FEDatas<Types...>()
    , integrate_values(false)
    , integrate_gradients(false)
    , evaluate_values(false)
    , evaluate_gradients(false)
    , evaluate_hessians(false)
    , initialized(false)
  {
    static_assert(FEData::max_degree == FEDatas::max_degree,
                  "The maximum degree must be the same for all FiniteElements!");
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "You need to construct this with a FEData object!");
  }

  FEDatas(const FEDatas<FEData, Types...> &fe_datas)
    : FEDatas<Types...>()
    , integrate_values(false)
    , integrate_gradients(false)
    , evaluate_values(false)
    , evaluate_gradients(false)
    , evaluate_hessians(false)
    , initialized(false)
  {
    (void)fe_datas;
    Assert(!fe_datas.initialized, ExcNotImplemented());
  }

private:
  std::unique_ptr<typename FEData::FEEvaluationType> fe_evaluation = nullptr;
  bool integrate_values = false;
  bool integrate_gradients = false;
  bool evaluate_values = false;
  bool evaluate_gradients = false;
  bool evaluate_hessians = false;
  bool initialized = false;
};

template <class FEData1, class FEData2>
typename std::enable_if<CFL::Traits::is_fe_data<FEData1>::value, FEDatas<FEData1, FEData2>>::type operator, (const FEData1 &,
    const FEData2 &)
{
  static_assert(CFL::Traits::is_fe_data<FEData2>::value, "Only FEData objects can be added!");
  return FEDatas<FEData1, FEData2>();
}

#endif // FE_DATA_H
