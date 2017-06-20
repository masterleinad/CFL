#ifndef FE_DATA_H
#define FE_DATA_H

#include <cfl/dealii_matrixfree.h>
#include <cfl/traits.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

template <typename... Types>
class FEDatas;

template <template <int, int> class FiniteElementType, int fe_degree, int n_components, int dim,
          unsigned int fe_no, unsigned int max_fe_degree, typename Number = double>
class FEData final
{
public:
  using FEEvaluationType =
    typename dealii::FEEvaluation<dim, fe_degree, max_fe_degree + 1, n_components, Number>;
  using NumberType = Number;
  using TensorTraits = CFL::Traits::Tensor<(n_components > 1 ? 1 : 0), dim>;
  static constexpr unsigned int fe_number = fe_no;
  static constexpr unsigned int max_degree = max_fe_degree;
  const std::shared_ptr<const FiniteElementType<dim, dim>> fe;

  explicit FEData(const FiniteElementType<dim, dim>& fe_)
    : FEData(std::make_shared<const FiniteElementType<dim, dim>>(fe_))
  {
  }

  explicit FEData(std::shared_ptr<const FiniteElementType<dim, dim>> fe_)
    : fe(std::move(fe_))
  {
    static_assert(fe_degree <= max_degree, "fe_degree must not be greater than max_degree!");
    AssertThrow(fe->degree == fe_degree,
                dealii::ExcIndexRange(fe->degree, fe_degree, fe_degree + 1));
    AssertThrow(fe->n_components() == n_components,
                dealii::ExcDimensionMismatch(fe->n_components(), n_components));
  }

  bool
  evaluation_is_initialized() const
  {
    return fe_evaluation != nullptr;
  }

private:
  std::shared_ptr<FEEvaluationType> fe_evaluation = nullptr;

  template <typename... Types>
  friend class FEDatas;
};

template <template <int, int> class FiniteElementType, int fe_degree, int n_components, int dim,
          unsigned int fe_no, unsigned int max_fe_degree, typename Number = double>
class FEDataFace final
{
public:
  using FEEvaluationType =
    typename dealii::FEFaceEvaluation<dim, fe_degree, max_fe_degree + 1, n_components, Number>;
  using NumberType = Number;
  using TensorTraits = CFL::Traits::Tensor<(n_components > 1 ? 1 : 0), dim>;
  static constexpr unsigned int fe_number = fe_no;
  static constexpr unsigned int max_degree = max_fe_degree;
  const std::shared_ptr<const FiniteElementType<dim, dim>> fe;

  explicit FEDataFace(const FiniteElementType<dim, dim>& fe_)
    : FEDataFace(std::make_shared<const FiniteElementType<dim, dim>>(fe_))
  {
  }

  explicit FEDataFace(const std::shared_ptr<const FiniteElementType<dim, dim>> fe_)
    : fe(std::move(fe_))
  {
    static_assert(fe_degree <= max_degree, "fe_degree must not be greater than max_degree!");
    AssertThrow(fe->degree == fe_degree, dealii::ExcIndexRange(fe->degree, fe_degree, fe_degree));
    AssertThrow(fe->n_components() == n_components,
                dealii::ExcDimensionMismatch(fe->n_components(), n_components));
  }

  template <bool interior = true, bool exterior = true>
  bool
  evaluation_is_initialized() const
  {
    return ((interior || fe_evaluation_interior != nullptr) &&
            (exterior || fe_evaluation_exterior != nullptr));
  }

private:
  std::shared_ptr<FEEvaluationType> fe_evaluation_interior = nullptr;
  std::shared_ptr<FEEvaluationType> fe_evaluation_exterior = nullptr;

  template <typename... Types>
  friend class FEDatas;
};

namespace CFL::Traits
{
template <template <int, int> class FiniteElementType, int fe_degree, int n_components, int dim,
          unsigned int fe_no, unsigned int max_degree, typename Number>
struct is_fe_data<
  FEData<FiniteElementType, fe_degree, n_components, dim, fe_no, max_degree, Number>>
{
  static const bool value = true;
};

template <template <int, int> class FiniteElementType, int fe_degree, int n_components, int dim,
          unsigned int fe_no, unsigned int max_degree, typename Number>
struct is_fe_data_face<
  FEDataFace<FiniteElementType, fe_degree, n_components, dim, fe_no, max_degree, Number>>
{
  static const bool value = true;
};
} // namespace CFL::Traits

template <class FEData>
class FEDatas<FEData>
{
public:
  using FEDataType = FEData;
  using FEEvaluationType = typename FEData::FEEvaluationType;
  using TensorTraits = typename FEData::TensorTraits;
  using NumberType = typename FEData::NumberType;
  static constexpr unsigned int fe_number = FEData::fe_number;
  static constexpr unsigned int max_degree = FEData::max_degree;
  static constexpr unsigned int n = 1;

  // Note: This constructor is deliberately not marked as explicit to allow initializations like:
  // .......
  // FEData<....> fedata_obj;
  // FEDatas<decltype(fedata_obj)> fedatas_obj = fedata_obj;
  // .......
  FEDatas(const FEData fe_data_)
    : fe_data(std::move(fe_data_))
  {
    //    std::cout << "Constructor1" << std::endl;
    static_assert(CFL::Traits::is_fe_data<FEData>::value ||
                    CFL::Traits::is_fe_data_face<FEData>::value,
                  "You need to construct this with a FEData object!");
  }

  template <unsigned int fe_number_extern>
  static constexpr bool
  rank()
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return TensorTraits::rank;
  }

  template <class NewFEData>
  FEDatas<NewFEData, FEData>
  operator,(const NewFEData& new_fe_data) const
  {
    //    std::cout << "Constructor2" << std::endl;
    static_assert(CFL::Traits::is_fe_data<NewFEData>::value ||
                    CFL::Traits::is_fe_data_face<NewFEData>::value,
                  "Only FEData objects can be added!");

    return FEDatas<NewFEData, FEData>(new_fe_data, fe_data);
  }

  template <typename Cell>
  void
  reinit(const Cell& cell)
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Reinit cell FEDatas " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());
        fe_data.fe_evaluation->reinit(cell);
      }
  }

  template <typename Face>
  void
  reinit_face(const Face& face)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Reinit face FEDatas " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());
        fe_data.fe_evaluation_interior->reinit(face);
        fe_data.fe_evaluation_exterior->reinit(face);
      }
  }

  template <typename Face>
  void
  reinit_boundary(const Face& face)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Reinit face FEDatas " << fe_number << std::endl;
#endif
        // For boundaries we only consider interior faces
        Assert((fe_data.template evaluation_is_initialized<true, false>()),
               dealii::ExcInternalError());
        fe_data.fe_evaluation_interior->reinit(face);
      }
  }

  template <typename VectorType>
  void
  read_dof_values(const VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Read cell DoF values " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());

        if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
            fe_data.fe_evaluation->read_dof_values(vector.block(fe_number));
        else
          fe_data.fe_evaluation->read_dof_values(vector);
      }
  }

  template <typename VectorType>
  void
  read_dof_values_face(const VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Read face DoF values " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());

        if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
          {
            fe_data.fe_evaluation_interior->read_dof_values(vector.block(fe_number));
            fe_data.fe_evaluation_exterior->read_dof_values(vector.block(fe_number));
          }
        else
        {
          fe_data.fe_evaluation_interior->read_dof_values(vector);
          fe_data.fe_evaluation_exterior->read_dof_values(vector);
        }
      }
  }

  template <unsigned int fe_number_extern>
  void
  set_integration_flags(bool integrate_value, bool integrate_gradient)
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    integrate_values |= integrate_value;
    integrate_gradients |= integrate_gradient;
#ifdef DEBUG_OUTPUT
    std::cout << "integrate cell value end: " << fe_number << " " << integrate_values << " "
              << integrate_value << std::endl;
    std::cout << "integrate cell gradients end: " << fe_number << " " << integrate_gradients << " "
              << integrate_gradient << std::endl;
#endif
  }

  template <unsigned int fe_number_extern>
  void
  set_integration_flags_face(bool integrate_value, bool integrate_gradient)
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    integrate_values |= integrate_value;
    integrate_gradients |= integrate_gradient;
#ifdef DEBUG_OUTPUT
    std::cout << "integrate face value end: " << fe_number << " " << integrate_values << " "
              << integrate_value << std::endl;
    std::cout << "integrate face gradients end: " << fe_number << " " << integrate_gradients << " "
              << integrate_gradient << std::endl;
#endif
  }

  template <unsigned int fe_number_extern>
  void
  set_evaluation_flags(bool evaluate_value, bool evaluate_gradient, bool evaluate_hessian)
  {
    static_assert(CFL::Traits::is_fe_data<FEData>::value, "Component not found!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    evaluate_values |= evaluate_value;
    evaluate_gradients |= evaluate_gradient;
    evaluate_hessians |= evaluate_hessian;
  }

  template <unsigned int fe_number_extern>
  void
  set_evaluation_flags_face(bool evaluate_value, bool evaluate_gradient, bool evaluate_hessian)
  {
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value, "Component not found!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    evaluate_values |= evaluate_value;
    evaluate_gradients |= evaluate_gradient;
    evaluate_hessians |= evaluate_hessian;
  }

  template <typename VectorType>
  void
  distribute_local_to_global(VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value) if (integrate_values | integrate_gradients)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Distribute cell DoF values " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());
        if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
            fe_data.fe_evaluation->distribute_local_to_global(vector.block(fe_number));
        else
          fe_data.fe_evaluation->distribute_local_to_global(vector);
      }
  }

  template <typename VectorType>
  void
  distribute_local_to_global_face(VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value) if (integrate_values |
                                                                 integrate_gradients)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Distribute face DoF values " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());
        if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
          {
            fe_data.fe_evaluation_interior->distribute_local_to_global(vector.block(fe_number));
            fe_data.fe_evaluation_exterior->distribute_local_to_global(vector.block(fe_number));
          }
        else
        {
          fe_data.fe_evaluation_interior->distribute_local_to_global(vector);
          fe_data.fe_evaluation_exterior->distribute_local_to_global(vector);
        }
      }
  }

  template <int dim, typename OtherNumber>
  void
  initialize(const dealii::MatrixFree<dim, OtherNumber>& mf)
  {
    static_assert(std::is_same<NumberType, OtherNumber>::value,
                  "Number type of MatrixFree and FEDatas has to match!");

    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Initialize cell FEDatas " << fe_number << std::endl;
#endif
        fe_data.fe_evaluation.reset(new typename FEData::FEEvaluationType(mf, fe_number));
      }
    else
    {
#ifdef DEBUG_OUTPUT
      std::cout << "Initialize face FEDatas " << fe_number << std::endl;
#endif
      fe_data.fe_evaluation_interior.reset(new
                                           typename FEData::FEEvaluationType(mf, true, fe_number));
      fe_data.fe_evaluation_exterior.reset(new
                                           typename FEData::FEEvaluationType(mf, false, fe_number));
    }

    initialized = true;
  }

  void
  evaluate()
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Evaluate cell FEDatas " << fe_number << " " << evaluate_values << " "
                  << evaluate_gradients << " " << evaluate_hessians << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());

        fe_data.fe_evaluation->evaluate(evaluate_values, evaluate_gradients, evaluate_hessians);
      }
  }

  void
  evaluate_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Evaluate face FEDatas " << fe_number << " " << evaluate_values << " "
                  << evaluate_gradients << " " << evaluate_hessians << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());

        fe_data.fe_evaluation_interior->evaluate(evaluate_values, evaluate_gradients);
        fe_data.fe_evaluation_exterior->evaluate(evaluate_values, evaluate_gradients);
      }
  }

  template <unsigned int fe_number_extern = fe_number>
  static constexpr unsigned int
  get_n_q_points()
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value && fe_number_extern == fe_number)
      return FEData::FEEvaluationType::static_n_q_points;
    return 0;
  }

  template <unsigned int fe_number_extern = fe_number>
  static constexpr unsigned int
  get_n_q_points_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value && fe_number_extern == fe_number)
      return FEData::FEEvaluationType::static_n_q_points;
    return 0;
  }

  template <unsigned int fe_number_extern>
  auto
  get_gradient(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->get_gradient(q);
  }

  template <unsigned int fe_number_extern, bool interior>
  auto
  get_normal_gradient(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    if constexpr(interior) return fe_data.fe_evaluation_interior->get_normal_gradient(q);
    else
      return -(fe_data.fe_evaluation_exterior->get_normal_gradient(q));
  }

  template <unsigned int fe_number_extern>
  auto
  get_symmetric_gradient(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get symmetric gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->get_symmetric_gradient(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_divergence(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get divergence FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->get_divergence(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_laplacian(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get laplacian FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->get_laplacian(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_hessian_diagonal(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get hessian_diagonal FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->get_hessian_diagonal(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_hessian(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get hessian FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->get_hessian(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_value(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get value FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "This function can only be called for FEData objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->get_value(q);
  }

  template <unsigned int fe_number_extern, bool interior>
  auto
  get_face_value(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get face value FEDatas " << fe_number << " " << q << " " << interior << ": ";
#endif
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value,
                  "This function can only be called for FEDataFace objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    if constexpr(interior)
      {
        const auto value = fe_data.fe_evaluation_interior->get_value(q);
#ifdef DEBUG_OUTPUT
        std::cout << value[0] << " " << value[1] << std::endl;
#endif
        return value;
      }
    else
    {
      const auto value = fe_data.fe_evaluation_exterior->get_value(q);
#ifdef DEBUG_OUTPUT
      std::cout << value[0] << " " << value[1] << std::endl;
#endif
      return value;
    }
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_gradient(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "This function can only be called for FEData objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_data.fe_evaluation->submit_gradient(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_curl(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit curl FEDatas " << fe_number << " " << q << std::endl;
    for (unsigned int i = 0; i < 2; ++i)
      std::cout << " " << value[i][0];
    std::cout << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "This function can only be called for FEData objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_data.fe_evaluation->submit_curl(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_divergence(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit divergence FEDatas " << fe_number << " " << q << std::endl;
    for (unsigned int i = 0; i < 2; ++i)
      std::cout << " " << value[i][0];
    std::cout << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "This function can only be called for FEData objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_data.fe_evaluation->submit_divergence(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_symmetric_gradient(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit symmetric gradient FEDatas " << fe_number << " " << q << std::endl;
    for (unsigned int i = 0; i < 2; ++i)
      std::cout << " " << value[i][0];
    std::cout << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "This function can only be called for FEData objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_data.fe_evaluation->submit_symmetric_gradient(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_value(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit value FEDatas" << fe_number << " " << q << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "This function can only be called for FEData objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_data.fe_evaluation->submit_value(value, q);
  }

  template <unsigned int fe_number_extern, bool interior, typename ValueType>
  void
  submit_face_value(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit face value FEDatas" << fe_number << " " << q << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value,
                  "This function can only be called for FEDataFace objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    if constexpr(interior) fe_data.fe_evaluation_interior->submit_value(value, q);
    else
      fe_data.fe_evaluation_exterior->submit_value(value, q);
  }

  template <unsigned int fe_number_extern, bool interior, typename ValueType>
  void
  submit_normal_gradient(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit normal gradient FEDatas" << fe_number << " " << q << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value,
                  "This function can only be called for FEDataFace objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    if constexpr(interior) fe_data.fe_evaluation_interior->submit_normal_gradient(value, q);
    else
      fe_data.fe_evaluation_exterior->submit_normal_gradient(-value, q);
  }

  void
  integrate()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "integrate cell FEDatas " << fe_number << " " << integrate_values << " "
              << integrate_gradients << std::endl;
#endif
    if constexpr(CFL::Traits::is_fe_data<FEData>::value) if (integrate_values | integrate_gradients)
        fe_data.fe_evaluation->integrate(integrate_values, integrate_gradients);
  }

  void
  integrate_face()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "integrate face FEDatas " << fe_number << " " << integrate_values << " "
              << integrate_gradients << std::endl;
#endif
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value) if (integrate_values |
                                                                 integrate_gradients)
      {
        fe_data.fe_evaluation_interior->integrate(integrate_values, integrate_gradients);
        fe_data.fe_evaluation_exterior->integrate(integrate_values, integrate_gradients);
      }
  }

  template <unsigned int fe_number_extern>
  const auto&
  get_fe_data() const
  {
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "Component not found, not a FEData object!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data;
  }

  template <unsigned int fe_number_extern>
  const auto&
  get_fe_data_face() const
  {
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value,
                  "Component not found, not a FEDataFace object");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data;
  }

  template <unsigned int fe_number_extern>
  unsigned int
  dofs_per_cell() const
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->dofs_per_cell;
  }

  template <unsigned int fe_number_extern>
  static constexpr unsigned int
  tensor_dofs_per_cell()
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return FEData::FEEvaluationType::tensor_dofs_per_cell;
  }

  template <unsigned int fe_number_extern>
  auto
  begin_dof_values() const
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->begin_dof_values();
  }

protected:
  FEData fe_data;

  template <unsigned int fe_number_extern>
  void
  check_uniqueness()
  {
    static_assert(!CFL::Traits::is_fe_data<FEData>::value || fe_number != fe_number_extern,
                  "The fe_numbers have to be unique!");
  }

  template <unsigned int fe_number_extern>
  void
  check_uniqueness_face()
  {
    static_assert(!CFL::Traits::is_fe_data_face<FEData>::value || fe_number != fe_number_extern,
                  "The fe_numbers have to be unique!");
  }

private:
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
  using FEEvaluationType = typename FEData::FEEvaluationType;
  using TensorTraits = typename FEData::TensorTraits;
  using NumberType = typename FEData::NumberType;
  using Base = FEDatas<Types...>;
  static constexpr unsigned int fe_number = FEData::fe_number;
  static constexpr unsigned int max_degree = Base::max_degree;
  static constexpr unsigned int n = Base::n + 1;

  FEDatas(const FEData fe_data_, const FEDatas<Types...> fe_datas_)
    : FEDatas<Types...>(std::move(fe_datas_))
    , fe_data(std::move(fe_data_))
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
        Base::template check_uniqueness<fe_number>();
    else
      Base::template check_uniqueness_face<fe_number>();
    //    std::cout << "Constructor4" << std::endl;
    static_assert(FEData::max_degree == FEDatas::max_degree,
                  "The maximum degree must be the same for all FiniteElements!");
    static_assert(CFL::Traits::is_fe_data<FEData>::value ||
                    CFL::Traits::is_fe_data_face<FEData>::value,
                  "You need to construct this with a FEData object!");
  }

  explicit FEDatas(const FEData fe_data_, const Types... fe_datas_)
    : Base(fe_datas_...)
    , fe_data(std::move(fe_data_))
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
        Base::template check_uniqueness<fe_number>();
    else
      Base::template check_uniqueness_face<fe_number>();
    //    std::cout << "Constructor3" << std::endl;
    static_assert(FEData::max_degree == FEDatas::max_degree,
                  "The maximum degree must be the same for all FiniteElements!");
    static_assert(CFL::Traits::is_fe_data<FEData>::value ||
                    CFL::Traits::is_fe_data_face<FEData>::value,
                  "You need to construct this with a FEData object!");
  }

  template <unsigned int fe_number_extern>
  static constexpr unsigned int
  rank()
  {
    if constexpr(fe_number == fe_number_extern) { return TensorTraits::rank; }
    else
      return Base::template rank<fe_number_extern>();
  }

  template <int dim, typename OtherNumber>
  void
  initialize(const dealii::MatrixFree<dim, OtherNumber>& mf)
  {
    static_assert(std::is_same<NumberType, OtherNumber>::value,
                  "Number type of MatrixFree and FEDatas do not match!");

    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Initialize cell FEDatas " << fe_number << std::endl;
#endif
        fe_data.fe_evaluation.reset(new typename FEData::FEEvaluationType(mf, fe_number));
      }
    else
    {
#ifdef DEBUG_OUTPUT
      std::cout << "Initialize face FEDatas " << fe_number << std::endl;
#endif
      fe_data.fe_evaluation_interior.reset(new
                                           typename FEData::FEEvaluationType(mf, fe_number, true));
      fe_data.fe_evaluation_exterior.reset(new
                                           typename FEData::FEEvaluationType(mf, fe_number, false));
    }
    initialized = true;
    Base::initialize(mf);
  }

  template <typename Cell>
  void
  reinit(const Cell& cell)
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Reinit cell FEDatas " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());
        fe_data.fe_evaluation->reinit(cell);
      }
    Base::reinit(cell);
  }

  template <typename Face>
  void
  reinit_face(const Face& face)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Reinit face FEDatas " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());
        fe_data.fe_evaluation_interior->reinit(face);
        fe_data.fe_evaluation_exterior->reinit(face);
      }
    Base::reinit_face(face);
  }

  template <typename Face>
  void
  reinit_boundary(const Face& face)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Reinit face FEDatas " << fe_number << std::endl;
#endif
        // For boundaries we only consider the interior faces
        Assert((fe_data.template evaluation_is_initialized<true, false>()),
               dealii::ExcInternalError());
        fe_data.fe_evaluation_interior->reinit(face);
      }
    Base::reinit_boundary(face);
  }

  template <typename VectorType>
  void
  read_dof_values(const VectorType& vector)
  {
    if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Read DoF values " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());
        if constexpr(CFL::Traits::is_fe_data<FEData>::value)
            fe_data.fe_evaluation->read_dof_values(vector.block(fe_number));
        else
        {
          fe_data.fe_evaluation_interior->read_dof_values(vector.block(fe_number));
          fe_data.fe_evaluation_exterior->read_dof_values(vector.block(fe_number));
        }
        Base::read_dof_values(vector);
      }
    else
    {
      // TODO(darndt): something tries to instantiate this even for valid code. Find out who!
      AssertThrow(false, dealii::ExcNotImplemented());
      /*static_assert(CFL::Traits::is_block_vector<VectorType>::value,
                    "It only makes sense to have multiple FEData objects if "
                    "you provide a block vector.");*/
    }
  }

  template <typename VectorType>
  void
  distribute_local_to_global(VectorType& vector)
  {
    if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
      {
        if (integrate_values | integrate_gradients)
        {
#ifdef DEBUG_OUTPUT
          std::cout << "Distribute cell DoF values " << fe_number << std::endl;
#endif
          Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());
          if constexpr(CFL::Traits::is_fe_data<FEData>::value)
              fe_data.fe_evaluation->distribute_local_to_global(vector.block(fe_number));
          else
          {
            fe_data.fe_evaluation_interior->distribute_local_to_global(vector.block(fe_number));
            fe_data.fe_evaluation_exterior->distribute_local_to_global(vector.block(fe_number));
          }
        }
        Base::distribute_local_to_global(vector);
      }
    else
    {
      // TODO(darndt): something tries to instantiate this even for valid code. Find out who!
      AssertThrow(false, dealii::ExcNotImplemented());
      /*static_assert(CFL::Traits::is_block_vector<VectorType>::value,
                    "It only makes sense to have multiple FEData objects if "
                    "you provide a block vector.");*/
    }
  }

  void
  evaluate()
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Evaluate cell FEDatas " << fe_number << " " << evaluate_values << " "
                  << evaluate_gradients << " " << evaluate_hessians << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());
        fe_data.fe_evaluation->evaluate(evaluate_values, evaluate_gradients, evaluate_hessians);
      }
    Base::evaluate();
  }

  void
  evaluate_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Evaluate face FEDatas " << fe_number << " " << evaluate_values << " "
                  << evaluate_gradients << " " << evaluate_hessians << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), dealii::ExcInternalError());

        fe_data.fe_evaluation_interior->evaluate(evaluate_values, evaluate_gradients);
        fe_data.fe_evaluation_exterior->evaluate(evaluate_values, evaluate_gradients);
      }
    Base::evaluate_face();
  }

  void
  integrate()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "integrate cell FEDatas " << fe_number << " " << integrate_values << " "
              << integrate_gradients << std::endl;
#endif
    if constexpr(CFL::Traits::is_fe_data<FEData>::value) if (integrate_values | integrate_gradients)
        fe_data.fe_evaluation->integrate(integrate_values, integrate_gradients);
    Base::integrate();
  }

  void
  integrate_face()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "integrate face FEDatas " << fe_number << " " << integrate_values << " "
              << integrate_gradients << std::endl;
#endif
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value) if (integrate_values |
                                                                 integrate_gradients)
      {
        fe_data.fe_evaluation_interior->integrate(integrate_values, integrate_gradients);
        fe_data.fe_evaluation_exterior->integrate(integrate_values, integrate_gradients);
      }
    Base::integrate_face();
  }

  template <unsigned int fe_number_extern>
  void
  set_integration_flags(bool integrate_value, bool integrate_gradient)
  {
    if constexpr(fe_number == fe_number_extern)
      {
        integrate_values |= integrate_value;
        integrate_gradients |= integrate_gradient;
#ifdef DEBUG_OUTPUT
        std::cout << "integrate cell value: " << fe_number << " " << integrate_values << " "
                  << integrate_value << std::endl;
        std::cout << "integrate cell gradients: " << fe_number << " " << integrate_gradients << " "
                  << integrate_gradient << std::endl;
#endif
      }
    else
      Base::template set_integration_flags<fe_number_extern>(integrate_value, integrate_gradient);
  }

  template <unsigned int fe_number_extern>
  void
  set_evaluation_flags(bool evaluate_value, bool evaluate_gradient, bool evaluate_hessian)
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value && fe_number == fe_number_extern)
      {
        evaluate_values |= evaluate_value;
        evaluate_gradients |= evaluate_gradient;
        evaluate_hessians |= evaluate_hessian;
      }
    else
    {
      Base::template set_evaluation_flags<fe_number_extern>(
        evaluate_value, evaluate_gradient, evaluate_hessian);
    }
  }

  template <unsigned int fe_number_extern>
  void
  set_evaluation_flags_face(bool evaluate_value, bool evaluate_gradient, bool evaluate_hessian)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value && fe_number == fe_number_extern)
      {
        evaluate_values |= evaluate_value;
        evaluate_gradients |= evaluate_gradient;
        evaluate_hessians |= evaluate_hessian;
      }
    else
    {
      Base::template set_evaluation_flags_face<fe_number_extern>(
        evaluate_value, evaluate_gradient, evaluate_hessian);
    }
  }

  template <unsigned int fe_number_extern = fe_number>
  static constexpr unsigned int
  get_n_q_points()
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value &&
                fe_number_extern == fe_number) return FEData::FEEvaluationType::static_n_q_points;
    else
      return Base::template get_n_q_points<fe_number_extern>();
  }

  template <unsigned int fe_number_extern = fe_number>
  static constexpr unsigned int
  get_n_q_points_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value &&
                fe_number_extern == fe_number) return FEData::FEEvaluationType::static_n_q_points;
    else
      return Base::template get_n_q_points_face<fe_number_extern>();
  }

  template <unsigned int fe_number_extern>
  auto
  get_gradient(unsigned int q) const
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "get gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
        return fe_data.fe_evaluation->get_gradient(q);
      }
    else
      return Base::template get_gradient<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern, bool interior>
  auto
  get_normal_gradient(unsigned int q) const
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "get normal gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
        if constexpr(interior) return fe_data.fe_evaluation_interior->get_normal_gradient(q);
        else
          return -(fe_data.fe_evaluation_exterior->get_normal_gradient(q));
      }
    else
      return Base::template get_normal_gradient<fe_number_extern, interior>(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_symmetric_gradient(unsigned int q) const
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "get symmetric gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
        return fe_data.fe_evaluation->get_symmetric_gradient(q);
      }
    else
      return Base::template get_symmetric_gradient<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_divergence(unsigned int q) const
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "get divergence FEDatas " << fe_number << " " << q << std::endl;
#endif
        return fe_data.fe_evaluation->get_divergence(q);
      }
    else
      return Base::template get_divergence<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_laplacian(unsigned int q) const
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "get laplacian FEDatas " << fe_number << " " << q << std::endl;
#endif
        return fe_data.fe_evaluation->get_laplacian(q);
      }
    else
      return Base::template get_laplacian<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_hessian_diagonal(unsigned int q) const
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "get hessian_diagonal FEDatas " << fe_number << " " << q << std::endl;
#endif
        return fe_data.fe_evaluation->get_hessian_diagonal(q);
      }
    else
      return Base::template get_hessian_diagonal<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_hessian(unsigned int q) const
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "get hessian FEDatas " << fe_number << " " << q << std::endl;
#endif
        return fe_data.fe_evaluation->get_hessian(q);
      }
    else
      return Base::template get_hessian<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern>
  auto
  get_value(unsigned int q) const
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "get value FEDatas " << fe_number << " " << q << std::endl;
#endif
        return fe_data.fe_evaluation->get_value(q);
      }
    else
      return Base::template get_value<fe_number_extern>(q);
  }

  template <unsigned int fe_number_extern, bool interior>
  auto
  get_face_value(unsigned int q) const
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "get face value FEDatas " << fe_number << " " << q << " " << interior
                  << ": ";

#endif
        if constexpr(interior)
          {
            const auto value = fe_data.fe_evaluation_interior->get_value(q);
#ifdef DEBUG_OUTPUT
            std::cout << value[0] << " " << value[1] << std::endl;
#endif
            return value;
          }
        else
        {
          const auto value = fe_data.fe_evaluation_exterior->get_value(q);
#ifdef DEBUG_OUTPUT
          std::cout << value[0] << " " << value[1] << std::endl;
#endif
          return value;
        }
      }
    else
      return Base::template get_face_value<fe_number_extern, interior>(q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_curl(const ValueType& value, unsigned int q)
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "submit curl FEDatas " << fe_number << " " << q << std::endl;
#endif
        fe_data.fe_evaluation->submit_curl(value, q);
      }
    else
      Base::template submit_curl<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_divergence(const ValueType& value, unsigned int q)
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "submit divergence FEDatas " << fe_number << " " << q << std::endl;
#endif
        fe_data.fe_evaluation->submit_divergence(value, q);
      }
    else
      Base::template submit_divergence<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_symmetric_gradient(const ValueType& value, unsigned int q)
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "submit symmetric gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
        fe_data.fe_evaluation->submit_symmetric_gradient(value, q);
      }
    else
      Base::template submit_symmetric_gradient<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_gradient(const ValueType& value, unsigned int q)
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "submit gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
        fe_data.fe_evaluation->submit_gradient(value, q);
      }
    else
      Base::template submit_gradient<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_value(const ValueType& value, unsigned int q)
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "submit value FEDatas " << fe_number << " " << q << std::endl;
#endif
        fe_data.fe_evaluation->submit_value(value, q);
      }
    else
      Base::template submit_value<fe_number_extern, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, bool interior, typename ValueType>
  void
  submit_face_value(const ValueType& value, unsigned int q)
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "submit face value FEDatas" << fe_number << " " << q << std::endl;
#endif
        if constexpr(interior) fe_data.fe_evaluation_interior->submit_value(value, q);
        else
          fe_data.fe_evaluation_exterior->submit_value(value, q);
      }
    else
      Base::template submit_face_value<fe_number_extern, interior, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern, bool interior, typename ValueType>
  void
  submit_normal_gradient(const ValueType& value, unsigned int q)
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "submit normal gradient" << fe_number << " " << q << std::endl;
#endif
        if constexpr(interior) fe_data.fe_evaluation_interior->submit_normal_gradient(value, q);
        else
          fe_data.fe_evaluation_exterior->submit_normal_gradient(-value, q);
      }
    else
      Base::template submit_normal_gradient<fe_number_extern, interior, ValueType>(value, q);
  }

  template <unsigned int fe_number_extern>
  const auto&
  get_fe_data() const
  {
    if constexpr(fe_number == fe_number_extern) return fe_data;
    else
      return Base::template get_fe_data<fe_number_extern>();
  }

  template <unsigned int fe_number_extern>
  const auto&
  get_fe_data_face() const
  {
    if constexpr(fe_number == fe_number_extern &&
                CFL::Traits::is_fe_data_face<FEData>::value) return fe_data;
    else
      return Base::template get_fe_data_face<fe_number_extern>();
  }

  template <unsigned int fe_number_extern>
  unsigned int
  dofs_per_cell() const
  {
    if constexpr(fe_number == fe_number_extern) { return fe_data.fe_evaluation->dofs_per_cell; }
    else
      return Base::template dofs_per_cell<fe_number_extern>();
  }

  template <unsigned int fe_number_extern>
  static constexpr unsigned int
  tensor_dofs_per_cell()
  {
    if constexpr(fe_number ==
                fe_number_extern) return FEData::FEEvaluationType::tensor_dofs_per_cell;
    else
      return Base::template tensor_dofs_per_cell<fe_number_extern>();
  }

  template <unsigned int fe_number_extern>
  const auto&
  begin_dof_values() const
  {
    if constexpr(fe_number == fe_number_extern) { return fe_data.fe_evaluation->begin_dof_values(); }
    else
      return Base::template begin_dof_values<fe_number_extern>();
  }

  template <class FEDataOther>
  typename std::enable_if_t<CFL::Traits::is_fe_data<FEDataOther>::value ||
                              CFL::Traits::is_fe_data_face<FEDataOther>::value,
                            FEDatas<FEDataOther, FEData, Types...>>
  operator,(const FEDataOther& new_fe_data) const
  {
    return FEDatas<FEDataOther, FEData, Types...>(new_fe_data, *this);
  }

protected:
  FEData fe_data;

  template <unsigned int fe_number_extern>
  void
  check_uniqueness()
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value) static_assert(
        fe_number != fe_number_extern, "The fe_numbers have to be unique!");
    Base::template check_uniqueness<fe_number_extern>();
  }

  template <unsigned int fe_number_extern>
  void
  check_uniqueness_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value) static_assert(
        fe_number != fe_number_extern, "The fe_numbers have to be unique!");
    Base::template check_uniqueness_face<fe_number_extern>();
  }

private:
  bool integrate_values = false;
  bool integrate_gradients = false;
  bool evaluate_values = false;
  bool evaluate_gradients = false;
  bool evaluate_hessians = false;
  bool initialized = false;
};

template <class FEData, typename... Types>
typename std::enable_if_t<CFL::Traits::is_fe_data<FEData>::value ||
                            CFL::Traits::is_fe_data_face<FEData>::value,
                          FEDatas<FEData, Types...>>
operator,(const FEData& new_fe_data, const FEDatas<Types...>& old_fe_data)
{
  return old_fe_data.operator,(new_fe_data);
}

template <class FEData1, class FEData2>
std::enable_if_t<CFL::Traits::is_fe_data<FEData1>::value ||
                   CFL::Traits::is_fe_data_face<FEData1>::value,
                 FEDatas<FEData1, FEData2>>
operator,(const FEData1& fe_data1, const FEData2& fe_data2)
{
  static_assert(CFL::Traits::is_fe_data<FEData2>::value ||
                  CFL::Traits::is_fe_data_face<FEData2>::value,
                "Only FEData objects can be added!");
  return FEDatas<FEData1, FEData2>(fe_data1, fe_data2);
}

#endif // FE_DATA_H
