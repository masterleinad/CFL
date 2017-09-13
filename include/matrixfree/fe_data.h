#ifndef DEALII_MATRIXFREE_FE_DATA_H
#define DEALII_MATRIXFREE_FE_DATA_H

#include <cfl/fefunctions.h>
#include <cfl/traits.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace CFL::dealii::MatrixFree
{
template <typename... Types>
class FEDatas;

/**
* @brief Class to hold Finite Element and the associated FEEvaluation
*
* A MatrixFree implementation requires FiniteElement on reference cell,
* MatrixFree for collection of needed data and FEEvaluation for actual
* evaluation. The FEData class is a provision in CFL library which
* holds the Finite Element and saves the FEEvaluation which is appropriate
* for this Finite Element.
* In order to be usable, the \ref FEData object must be part of \ref FEDatas
* which is a container for such objects.
* This class has been designed in such a way that the description of shape
* functions has to be provided at compile time but the decision for which
* FEEvaluation has to be used for actual evaluation is left to run-time.
* This has the obvious advantage of using different schemes for evaluation
* on a mesh while having the same FiniteElement defined on the reference cell.
* This distinction can be seen in the template parameter
* <code>FiniteElementType<\code> which implies compile time dependency vs the
* member variable <code> fe_evaluation <\code> which is initialized at run-time
* when appropriate.
* It is important to note that every finite element which is added to \ref
* FEDatas should be uniquely identifiable. For this reason the class
* definition of \FEData requires <code>fe_no<\code>
*
* For more on usage of this class, refer \ref FEDatas
*
* <h3> Usage example </h3>
* <code>
* 	  const auto fe_shared = std::make_shared<FE_Q<2>>(2);
      FEData<FE_Q, 2, 1, 2, 0, 2, double> fedata_e_system(fe_shared);
      FEData<FE_Q, 2, 1, 2, 1, 2, double> fedata_u_system(fe_shared);
* </code>
*
*
*/
template <template <int, int> class FiniteElementType, int fe_degree, int n_components, int dim,
          unsigned int fe_no, unsigned int max_fe_degree, typename Number = double>
class FEData final
{
public:
  using FEEvaluationType =
    typename ::dealii::FEEvaluation<dim, fe_degree, max_fe_degree + 1, n_components, Number>;
  using NumberType = Number;
  using TensorTraits = CFL::Traits::Tensor<(n_components > 1 ? 1 : 0), dim>;
  static constexpr unsigned int fe_number = fe_no;
  static constexpr unsigned int max_degree = max_fe_degree;
  const std::shared_ptr<const FiniteElementType<dim, dim>> fe;

  /**
   * Explicit constructor for providing the shape function description
   * which is a reference to an object of any class derived from
   * \ref FiniteElement
   * \note The caller should ensure that the provided object is not
   * destroyed till the life-time of FEData, typically throughout the
   * life time of the program.
   */
  explicit FEData(const FiniteElementType<dim, dim>& fe_)
    : FEData(std::make_shared<const FiniteElementType<dim, dim>>(fe_))
  {
  }

  /**
   * Explicit constructor for providing the shape function description
   * which is an object of any class derived from \ref FiniteElement.
   * For efficiency reasons, shared_pointer to such an object can be provided
   */
  explicit FEData(std::shared_ptr<const FiniteElementType<dim, dim>> fe_)
    : fe(std::move(fe_))
  {
    static_assert(fe_degree <= max_degree, "fe_degree must not be greater than max_degree!");
    AssertThrow(fe->degree == fe_degree,
                ::dealii::ExcIndexRange(fe->degree, fe_degree, fe_degree + 1));
    AssertThrow(fe->n_components() == n_components,
                ::dealii::ExcDimensionMismatch(fe->n_components(), n_components));
  }

  /**
   * Returns True if the FEData object has been initialized with FEEvaluation
   * object. This function is public for it to be accessible to FEDatas which
   * is a container for FEData
   */
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

/**
 * Similar in design to \ref FEData class. This class however is designed for
 * evaluation over faces - whether interior faces or exterior (boundary) faces
 */
template <template <int, int> class FiniteElementType, int fe_degree, int n_components, int dim,
          unsigned int fe_no, unsigned int max_fe_degree, typename Number = double>
class FEDataFace final
{
public:
  using FEEvaluationType =
    typename ::dealii::FEFaceEvaluation<dim, fe_degree, max_fe_degree + 1, n_components, Number>;
  using NumberType = Number;
  using TensorTraits = CFL::Traits::Tensor<(n_components > 1 ? 1 : 0), dim>;
  static constexpr unsigned int fe_number = fe_no;
  static constexpr unsigned int max_degree = max_fe_degree;
  const std::shared_ptr<const FiniteElementType<dim, dim>> fe;

  /**
   * Similar in design to \ref FEData
   */
  explicit FEDataFace(const FiniteElementType<dim, dim>& fe_)
    : FEDataFace(std::make_shared<const FiniteElementType<dim, dim>>(fe_))
  {
  }

  /**
   * Similar in design to \ref FEData
   */
  explicit FEDataFace(const std::shared_ptr<const FiniteElementType<dim, dim>> fe_)
    : fe(std::move(fe_))
  {
    static_assert(fe_degree <= max_degree, "fe_degree must not be greater than max_degree!");
    AssertThrow(fe->degree == fe_degree, ::dealii::ExcIndexRange(fe->degree, fe_degree, fe_degree));
    AssertThrow(fe->n_components() == n_components,
                ::dealii::ExcDimensionMismatch(fe->n_components(), n_components));
  }

  /**
   * Similar in design to \ref FEData. However, in here we check if
   * initialization has been done for both interior and exterior faces
   */
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
}

namespace CFL::Traits
{
/**
 * @brief Trait to determine if a given type is CFL FEData
 * @todo Should be moved to dealii_matrixfree.h?
 *
 */
template <template <int, int> class FiniteElementType, int fe_degree, int n_components, int dim,
          unsigned int fe_no, unsigned int max_degree, typename Number>
struct is_fe_data<CFL::dealii::MatrixFree::FEData<FiniteElementType, fe_degree, n_components, dim,
                                                  fe_no, max_degree, Number>>
{
  static const bool value = true;
};

/**
 * @brief Trait to determine if a given type is CFL FEDataFace
 * @todo Should be moved to dealii_matrixfree.h?
 *
 */
template <template <int, int> class FiniteElementType, int fe_degree, int n_components, int dim,
          unsigned int fe_no, unsigned int max_degree, typename Number>
struct is_fe_data_face<CFL::dealii::MatrixFree::FEDataFace<
  FiniteElementType, fe_degree, n_components, dim, fe_no, max_degree, Number>>
{
  static const bool value = true;
};
} // namespace CFL::Traits

namespace CFL::dealii::MatrixFree
{
/**
* @brief Class to provide FEEvaluation services in the scope of CFL
*
* Also refer to \ref FEData class.
* FEDatas is a static container class which allows to build collection of
* \ref FEData and/or \ref FEDataFace objects at compile time.
* Further, it provides member functions which are a wrapper around the
* services of \ref FEEvaluation class.
* These member functions are themselves template functions and are
* templatized with unique <code> fe_no <\code> which is a necessity
* for storing these items in the collection
*
* <h3> What is static container </h3>
* First of all, this is not a standard terminology!
* We define static container to be a collection object whose length
* and the objects which it contains are defined at compile time. The contents
* of the objects can be modified at runtime, but new objects can not be
* added nor old objects be deleted during runtime. Think of it as a mid way
* between array and a vector - like an array its memory requirements are fixed
* at compile time; like a vector it can grow, but unlike a vector which grows
* dynamically the growth of a static container is only possible at compile time
*
* <h3> Usage example </h3>
* The items can be added to container using comma operation which is an
* overloaded operator in this class
*
* <code>
* 	  const auto fe_shared = std::make_shared<FE_Q<2>>(2);
      FEData<FE_Q, 2, 1, 2, 0, 2, double> fedata_e_system(fe_shared);
      FEData<FE_Q, 2, 1, 2, 1, 2, double> fedata_u_system(fe_shared);

       FEData<FE_Q, 2, 1, 2, 2, 2, double> fedata_x_system(fe_shared)
      auto fedatas_1 = (fedata_e_system, fedata_u_system);
      auto fedatas_2 = (fedatas_1, fedata_x_system);
* </code>
*
* <h3>Implementation</h3>
* The static container concept is implemented using C++ variadic templates.
* When the first time an FEDatas object is created using two FEDatas objects
* appended using comma, a heirarchy of base-derived classes is created as:
* <li> fedatas_1:
*  *   @verbatim
 *		FEDatas<FEData>  --> holds fedata_e_system
 *		     ^
 *		     |
 *		     |
 *		FEDatas<FEData,FEData> --> holds fedata_u_system
 *   @endverbatim
* Next time, when a new FEData object is added to existing FEDatas, the
* new heirarchy will look like:
** <li> fedatas_2:
*  *   @verbatim
 *		FEDatas<FEData>  --> holds fedata_e_system
 *		     ^
 *		     |
 *		     |
 *		FEDatas<FEData,FEData> --> holds fedata_u_system
 *		     ^
 *		     |
 *		     |
 *		FEDatas<FEData,FEData,FEData> --> holds fedata_x_system
 *   @endverbatim
*/
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

  /**
   * Constructor for storing the provided FEData object
   *

   * \note This constructor is deliberately not marked as explicit to allow initializations like:
          <code>
           .......
           FEData<....> fedata_obj;
           FEDatas<decltype(fedata_obj)> fedatas_obj = fedata_obj;
           .......
          <\code>
  */
  FEDatas(const FEData fe_data_)
    : fe_data(std::move(fe_data_))
  {
    //    std::cout << "Constructor1" << std::endl;
    static_assert(CFL::Traits::is_fe_data<FEData>::value ||
                    CFL::Traits::is_fe_data_face<FEData>::value,
                  "You need to construct this with a FEData object!");
  }

  /**
   * Returns the rank of the FEData object stored in this collection
   *
   */
  template <unsigned int fe_number_extern>
  static constexpr bool
  rank()
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return TensorTraits::rank;
  }

  /**
   * Comma operator to append another item to this collection in a
   * static manner
   *
   */
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

  /**
   * Wrapper around reinit function of FEEvaluation used for cell
   *
   */
  template <typename Cell>
  void
  reinit(const Cell& cell)
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Reinit cell FEDatas " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
        fe_data.fe_evaluation->reinit(cell);
      }
  }

  /**
   * Wrapper around reinit function of FEEvaluation used for faces
   *
   */
  template <typename Face>
  void
  reinit_face(const Face& face)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Reinit face FEDatas " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
        fe_data.fe_evaluation_interior->reinit(face);
        fe_data.fe_evaluation_exterior->reinit(face);
      }
  }

  /**
   * Wrapper around reinit function of FEEvaluation used for exterior face
   *
   */
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
               ::dealii::ExcInternalError());
        fe_data.fe_evaluation_interior->reinit(face);
      }
  }

  /**
   * Wrapper around read_dof_values function of FEEvaluation used for cell
   *
   */
  template <typename VectorType>
  void
  read_dof_values(const VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Read cell DoF values " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());

        if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
            fe_data.fe_evaluation->read_dof_values(vector.block(fe_number));
        else
          fe_data.fe_evaluation->read_dof_values(vector);
      }
  }

  /**
   * Wrapper around read_dof_values function of FEEvaluation used for face
   *
   */
  template <typename VectorType, bool interior = true, bool exterior = true>
  void
  read_dof_values_face(const VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
        Assert((fe_data.template evaluation_is_initialized<interior, exterior>()),
               ::dealii::ExcInternalError());
#ifdef DEBUG_OUTPUT
        std::cout << "Read face DoF values " << fe_number << " " << interior << " " << exterior
                  << std::endl;
#endif
        if constexpr(interior)
          {
            if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
                fe_data.fe_evaluation_interior->read_dof_values(vector.block(fe_number));
            else
              fe_data.fe_evaluation_interior->read_dof_values(vector);
          }
        if constexpr(exterior)
          {
            if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
                fe_data.fe_evaluation_exterior->read_dof_values(vector.block(fe_number));
            else
              fe_data.fe_evaluation_exterior->read_dof_values(vector);
          }
      }
  }

  /**
   * Saves the integration flags on cell used later for MatrixFree evaluation
   *
   */
  template <unsigned int fe_number_extern>
  void
  set_integration_flags(bool integrate_value, bool integrate_gradient)
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    static_assert(CFL::Traits::is_fe_data<FEData>::value, "Must be cell object!");
    integrate_values |= integrate_value;
    integrate_gradients |= integrate_gradient;
#ifdef DEBUG_OUTPUT
    std::cout << "integrate cell value: " << fe_number << " " << integrate_values << " "
              << integrate_value << std::endl;
    std::cout << "integrate cell gradients: " << fe_number << " " << integrate_gradients << " "
              << integrate_gradient << std::endl;
#endif
  }

  /**
   * Clears the integration flags
   *
   */
  void
  reset_integration_flags_face_and_boundary()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
        integrate_values = false;
        integrate_values_exterior = false;
        integrate_gradients = false;
        integrate_gradients_exterior = false;
      }
  }

  /**
   * Saves the integration flags on faces used later for MatrixFree evaluation
   *
   */
  template <unsigned int fe_number_extern>
  void
  set_integration_flags_face_and_boundary(bool integrate_value, bool integrate_value_exterior,
                                          bool integrate_gradient, bool integrate_gradient_exterior)
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value, "Must be face object!");
    integrate_values |= integrate_value;
    integrate_values_exterior |= integrate_value_exterior;
    integrate_gradients |= integrate_gradient;
    integrate_gradients_exterior |= integrate_gradient_exterior;
#ifdef DEBUG_OUTPUT
    std::cout << "integrate face value: " << fe_number << " " << integrate_values << " "
              << integrate_value_exterior << std::endl;
    std::cout << "integrate face value exterior: " << fe_number << " " << integrate_values_exterior
              << " " << integrate_value_exterior << std::endl;
    std::cout << "integrate face gradients: " << fe_number << " " << integrate_gradients << " "
              << integrate_gradient << std::endl;
    std::cout << "integrate face gradients exterior: " << fe_number << " "
              << integrate_gradients_exterior << " " << integrate_gradient_exterior << std::endl;
#endif
  }

  /**
   * Saves the integration flags on cell used later for MatrixFree evaluation
   *
   */
  template <unsigned int fe_number_extern>
  void
  set_evaluation_flags(bool evaluate_value, bool evaluate_gradient, bool evaluate_hessian)
  {
    static_assert(CFL::Traits::is_fe_data<FEData>::value, "Component not found!");
    static_assert(fe_number == fe_number_extern, "Must be cell object!");
    evaluate_values |= evaluate_value;
    evaluate_gradients |= evaluate_gradient;
    evaluate_hessians |= evaluate_hessian;
#ifdef DEBUG_OUTPUT
    std::cout << "evaluate cell value: " << fe_number << " " << evaluate_values << " "
              << evaluate_value << std::endl;
    std::cout << "evaluate cell gradients: " << fe_number << " " << evaluate_gradients << " "
              << evaluate_gradient << std::endl;
    std::cout << "evaluate cell hessian: " << fe_number << " " << evaluate_hessians << " "
              << evaluate_hessian << std::endl;
#endif
  }

  /**
   * Saves the integration flags on faces used later for MatrixFree evaluation
   *
   */
  template <unsigned int fe_number_extern>
  void
  set_evaluation_flags_face(bool evaluate_value, bool evaluate_gradient, bool evaluate_hessian)
  {
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value, "Must be face object!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    evaluate_values |= evaluate_value;
    evaluate_gradients |= evaluate_gradient;
    evaluate_hessians |= evaluate_hessian;
#ifdef DEBUG_OUTPUT
    std::cout << "evaluate face value: " << fe_number << " " << evaluate_values << " "
              << evaluate_value << std::endl;
    std::cout << "evaluate face gradients: " << fe_number << " " << evaluate_gradients << " "
              << evaluate_gradient << std::endl;
    std::cout << "evaluate face hessian: " << fe_number << " " << evaluate_hessians << " "
              << evaluate_hessian << std::endl;
#endif
  }

  /**
   * Wrapper around distribute_local_to_global function of FEEvaluation used for cell
   *
   */
  template <typename VectorType>
  void
  distribute_local_to_global(VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value) if (integrate_values | integrate_gradients)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Distribute cell DoF values " << fe_number << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
        if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
            fe_data.fe_evaluation->distribute_local_to_global(vector.block(fe_number));
        else
          fe_data.fe_evaluation->distribute_local_to_global(vector);
      }
  }

  /**
   * Wrapper around distribute_local_to_global function of FEEvaluation used for faces
   *
   */
  template <typename VectorType, bool interior = true, bool exterior = true>
  void
  distribute_local_to_global_face(VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {

        Assert((fe_data.template evaluation_is_initialized<interior, exterior>()),
               ::dealii::ExcInternalError());
#ifdef DEBUG_OUTPUT
        std::cout << "Distribute face DoF values " << fe_number << " " << interior << " "
                  << exterior << std::endl;
#endif
        if constexpr(interior) if (integrate_values | integrate_gradients)
          {
            if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
                fe_data.fe_evaluation_interior->distribute_local_to_global(vector.block(fe_number));
            else
              fe_data.fe_evaluation_interior->distribute_local_to_global(vector);
          }
        if constexpr(exterior) if (integrate_values_exterior | integrate_gradients_exterior)
          {
            if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
                fe_data.fe_evaluation_exterior->distribute_local_to_global(vector.block(fe_number));
            else
              fe_data.fe_evaluation_exterior->distribute_local_to_global(vector);
          }
      }
  }

  /**
   * Initializes the FEData object with appropriate FEEvaluation object
   *
   */
  template <int dim, typename OtherNumber>
  void
  initialize(const ::dealii::MatrixFree<dim, OtherNumber>& mf)
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

  /**
   * Wrapper around evaluate function of FEEvaluation used for cell
   *
   */
  void
  evaluate()
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Evaluate cell FEDatas " << fe_number << " " << evaluate_values << " "
                  << evaluate_gradients << " " << evaluate_hessians << std::endl;
#endif
        Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());

        fe_data.fe_evaluation->evaluate(evaluate_values, evaluate_gradients, evaluate_hessians);
      }
  }

  /**
   * Wrapper around evaluate function of FEEvaluation used for faces
   *
   */
  template <bool interior = true, bool exterior = true>
  void
  evaluate_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Evaluate face FEDatas " << fe_number << " " << evaluate_values << " "
                  << evaluate_gradients << " " << evaluate_hessians << std::endl;
#endif
        Assert((fe_data.template evaluation_is_initialized<interior, exterior>()),
               ::dealii::ExcInternalError());

        if constexpr(interior)
            fe_data.fe_evaluation_interior->evaluate(evaluate_values, evaluate_gradients);
        if constexpr(exterior)
            fe_data.fe_evaluation_exterior->evaluate(evaluate_values, evaluate_gradients);
      }
  }

  /**
   * Wrapper around static_n_q_points of FEEvaluation used for cell
   *
   */
  template <unsigned int fe_number_extern = fe_number>
  static constexpr unsigned int
  get_n_q_points()
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value &&
                fe_number_extern == fe_number) return FEData::FEEvaluationType::static_n_q_points;
    return 0;
  }

  /**
   * Wrapper around static_n_q_points of FEEvaluation used for faces
   *
   */
  template <unsigned int fe_number_extern = fe_number>
  static constexpr unsigned int
  get_n_q_points_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value &&
                fe_number_extern == fe_number) return FEData::FEEvaluationType::static_n_q_points;
    return 0;
  }

  /**
   * Wrapper around get_gradient function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around get_normal_gradient function of FEEvaluation
   *
   */
  template <unsigned int fe_number_extern, bool interior>
  auto
  get_normal_gradient(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get normal gradient FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(fe_number == fe_number_extern, "Component not found!");
    if constexpr(interior) return fe_data.fe_evaluation_interior->get_normal_gradient(q);
    else
      return -(fe_data.fe_evaluation_exterior->get_normal_gradient(q));
  }

  /**
   * Wrapper around get_symmetric_gradient function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around get_divergence function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around get_laplacian function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around get_hessian_diagonal function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around get_hessian function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around get_value function of FEEvaluation used for cell
   *
   */
  template <unsigned int fe_number_extern>
  auto
  get_value(unsigned int q) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "get value FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "This function can only be called for FEData objects!");
    Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
    static_assert(fe_number == fe_number_extern, "Component not found!");
    Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
    return fe_data.fe_evaluation->get_value(q);
  }

  /**
   * Wrapper around get_value function of FEEvaluation used for faces
   *
   */
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

  /**
   * Wrapper around submit_gradient function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around submit_curl function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around submit_divergence function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around submit_symmetric_gradient function of FEEvaluation
   *
   */
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

  /**
   * Wrapper around submit_value function of FEEvaluation used for cell
   *
   */
  template <unsigned int fe_number_extern, typename ValueType>
  void
  submit_value(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit value FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "This function can only be called for FEData objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    fe_data.fe_evaluation->submit_value(value, q);
  }

  /**
   * Wrapper around submit_value function of FEEvaluation used for faces
   *
   */
  template <unsigned int fe_number_extern, bool interior, typename ValueType>
  void
  submit_face_value(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit face value FEDatas " << fe_number << " " << q << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value,
                  "This function can only be called for FEDataFace objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    if constexpr(interior) fe_data.fe_evaluation_interior->submit_value(value, q);
    else
      fe_data.fe_evaluation_exterior->submit_value(value, q);
  }

  /**
   * Wrapper around submit_normal_gradient function of FEEvaluation
   *
   */
  template <unsigned int fe_number_extern, bool interior, typename ValueType>
  void
  submit_normal_gradient(const ValueType& value, unsigned int q)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "submit normal gradient " << fe_number << " " << q << std::endl;
#endif
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value,
                  "This function can only be called for FEDataFace objects!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    if constexpr(interior) fe_data.fe_evaluation_interior->submit_normal_gradient(value, q);
    else
      fe_data.fe_evaluation_exterior->submit_normal_gradient(-value, q);
  }

  /**
   * Wrapper around integrate function of FEEvaluation used for cell
   *
   */
  void
  integrate()
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
        if (integrate_values | integrate_gradients)
        {
#ifdef DEBUG_OUTPUT
          std::cout << "integrate cell FEDatas " << fe_number << " " << integrate_values << " "
                    << integrate_gradients << std::endl;
#endif
          fe_data.fe_evaluation->integrate(integrate_values, integrate_gradients);
        }
      }
  }

  /**
   * Wrapper around integrate function of FEEvaluation used for faces
   *
   */
  void
  integrate_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
        if (integrate_values | integrate_gradients)
        {
#ifdef DEBUG_OUTPUT
          std::cout << "integrate face FEDatas " << fe_number << " " << integrate_values << " "
                    << integrate_gradients << std::endl;
#endif
          fe_data.fe_evaluation_interior->integrate(integrate_values, integrate_gradients);
        }
        if (integrate_values_exterior | integrate_gradients_exterior)
        {
#ifdef DEBUG_OUTPUT
          std::cout << "integrate face exterior FEDatas " << fe_number << " "
                    << integrate_values_exterior << " " << integrate_gradients_exterior
                    << std::endl;
#endif
          fe_data.fe_evaluation_exterior->integrate(integrate_values_exterior,
                                                    integrate_gradients_exterior);
        }
      }
  }

  /**
   * Returns the FEData object with given fe_number
   *
   */
  template <unsigned int fe_number_extern>
  const auto&
  get_fe_data() const
  {
    static_assert(CFL::Traits::is_fe_data<FEData>::value,
                  "Component not found, not a FEData object!");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data;
  }

  /**
   * Returns the FEData object with given fe_number
   *
   */
  template <unsigned int fe_number_extern>
  const auto&
  get_fe_data_face() const
  {
    static_assert(CFL::Traits::is_fe_data_face<FEData>::value,
                  "Component not found, not a FEDataFace object");
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data;
  }

  /**
   * Returns the FEData object held with this class
   *
   */
  auto
  get_fe_datas() const
  {
    return fe_data;
  }

  /**
   * Wrapper around dofs_per_cell function of FEEvaluation
   *
   */
  template <unsigned int fe_number_extern>
  unsigned int
  dofs_per_cell() const
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->dofs_per_cell;
  }

  /**
   * Wrapper around tensor_dofs_per_cell of FEEvaluation
   *
   */
  template <unsigned int fe_number_extern>
  static constexpr unsigned int
  tensor_dofs_per_cell()
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return FEData::FEEvaluationType::tensor_dofs_per_cell;
  }

  /**
   * Wrapper around begin_dof_values function of FEEvaluation
   *
   */
  template <unsigned int fe_number_extern>
  auto
  begin_dof_values() const
  {
    static_assert(fe_number == fe_number_extern, "Component not found!");
    return fe_data.fe_evaluation->begin_dof_values();
  }

protected:
  FEData fe_data;

  /**
   * Ensures uniqueness of FEData objects added to the container
   *
   */
  template <unsigned int fe_number_extern>
  void
  check_uniqueness()
  {
    static_assert(!CFL::Traits::is_fe_data<FEData>::value || fe_number != fe_number_extern,
                  "The fe_numbers have to be unique!");
  }

  /**
   * Ensures uniqueness of FEData objects added to the container
   *
   */
  template <unsigned int fe_number_extern>
  void
  check_uniqueness_face()
  {
    static_assert(!CFL::Traits::is_fe_data_face<FEData>::value || fe_number != fe_number_extern,
                  "The fe_numbers have to be unique!");
  }

private:
  bool integrate_values = false;
  bool integrate_values_exterior = false;
  bool integrate_gradients = false;
  bool integrate_gradients_exterior = false;
  bool evaluate_values = false;
  bool evaluate_gradients = false;
  bool evaluate_hessians = false;
  bool initialized = false;
};

/**
 * @brief Class to provide FEEvaluation services in the scope of CFL
 * This is for variadic template definition of the FEDatas class.
 * Please refer to the documentation of the previous class
 */
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
    if constexpr(fe_number == fe_number_extern)
      return TensorTraits::rank;
    else
      return Base::template rank<fe_number_extern>();
  }

  template <int dim, typename OtherNumber>
  void
  initialize(const ::dealii::MatrixFree<dim, OtherNumber>& mf)
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
      fe_data.fe_evaluation_interior.reset(
        new typename FEData::FEEvaluationType{ mf, true, fe_number });
      fe_data.fe_evaluation_exterior.reset(
        new typename FEData::FEEvaluationType{ mf, false, fe_number });
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
        Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
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
        Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
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
               ::dealii::ExcInternalError());
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
        if constexpr(CFL::Traits::is_fe_data<FEData>::value)
          {
#ifdef DEBUG_OUTPUT
            std::cout << "Read cell DoF values " << fe_number << std::endl;
#endif
            Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
            fe_data.fe_evaluation->read_dof_values(vector.block(fe_number));
          }
        Base::read_dof_values(vector);
      }
    else
    {
      // TODO(darndt): something tries to instantiate this even for valid code. Find out who!
      AssertThrow(false, ::dealii::ExcNotImplemented());
      /*static_assert(CFL::Traits::is_block_vector<VectorType>::value,
                    "It only makes sense to have multiple FEData objects if "
                    "you provide a block vector.");*/
    }
  }

  template <typename VectorType, bool interior = true, bool exterior = true>
  void
  read_dof_values_face(const VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
        Assert((fe_data.template evaluation_is_initialized<interior, exterior>()),
               ::dealii::ExcInternalError());
#ifdef DEBUG_OUTPUT
        std::cout << "Read face DoF values " << fe_number << " " << interior << " " << exterior
                  << std::endl;
#endif
        if constexpr(interior)
          {
            if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
                fe_data.fe_evaluation_interior->read_dof_values(vector.block(fe_number));
            else
              fe_data.fe_evaluation_interior->read_dof_values(vector);
          }
        if constexpr(exterior)
          {
            if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
                fe_data.fe_evaluation_exterior->read_dof_values(vector.block(fe_number));
            else
              fe_data.fe_evaluation_exterior->read_dof_values(vector);
          }
      }
    Base::template read_dof_values_face<VectorType, interior, exterior>(vector);
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
          Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
          if constexpr(CFL::Traits::is_fe_data<FEData>::value)
              fe_data.fe_evaluation->distribute_local_to_global(vector.block(fe_number));
        }
        Base::distribute_local_to_global(vector);
      }
    else
    {
      // TODO(darndt): something tries to instantiate this even for valid code. Find out who!
      AssertThrow(false, ::dealii::ExcNotImplemented());
      /*static_assert(CFL::Traits::is_block_vector<VectorType>::value,
                    "It only makes sense to have multiple FEData objects if "
                    "you provide a block vector.");*/
    }
  }

  template <typename VectorType, bool interior = true, bool exterior = true>
  void
  distribute_local_to_global_face(VectorType& vector)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {

        Assert((fe_data.template evaluation_is_initialized<interior, exterior>()),
               ::dealii::ExcInternalError());
#ifdef DEBUG_OUTPUT
        std::cout << "Distribute face DoF values " << fe_number << " " << interior << " "
                  << exterior << std::endl;
#endif
        if constexpr(interior) if (integrate_values | integrate_gradients)
          {
            if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
                fe_data.fe_evaluation_interior->distribute_local_to_global(vector.block(fe_number));
            else
              fe_data.fe_evaluation_interior->distribute_local_to_global(vector);
          }
        if constexpr(exterior) if (integrate_values_exterior | integrate_gradients_exterior)
          {
            if constexpr(CFL::Traits::is_block_vector<VectorType>::value)
                fe_data.fe_evaluation_exterior->distribute_local_to_global(vector.block(fe_number));
            else
              fe_data.fe_evaluation_exterior->distribute_local_to_global(vector);
          }
      }
    Base::template distribute_local_to_global_face<VectorType, interior, exterior>(vector);
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
        Assert(fe_data.evaluation_is_initialized(), ::dealii::ExcInternalError());
        fe_data.fe_evaluation->evaluate(evaluate_values, evaluate_gradients, evaluate_hessians);
      }
    Base::evaluate();
  }

  template <bool interior = true, bool exterior = true>
  void
  evaluate_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
#ifdef DEBUG_OUTPUT
        std::cout << "Evaluate face FEDatas " << fe_number << " " << evaluate_values << " "
                  << evaluate_gradients << " " << evaluate_hessians << std::endl;
#endif
        Assert((fe_data.template evaluation_is_initialized<interior, exterior>()),
               ::dealii::ExcInternalError());

        if constexpr(interior)
            fe_data.fe_evaluation_interior->evaluate(evaluate_values, evaluate_gradients);
        if constexpr(exterior)
            fe_data.fe_evaluation_exterior->evaluate(evaluate_values, evaluate_gradients);
      }

    Base::template evaluate_face<interior, exterior>();
  }

  void
  integrate()
  {
    if constexpr(CFL::Traits::is_fe_data<FEData>::value)
      {
        if (integrate_values | integrate_gradients)
        {
#ifdef DEBUG_OUTPUT
          std::cout << "integrate cell FEDatas " << fe_number << " " << integrate_values << " "
                    << integrate_gradients << std::endl;
#endif
          fe_data.fe_evaluation->integrate(integrate_values, integrate_gradients);
        }
      }
    Base::integrate();
  }

  void
  integrate_face()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
        if (integrate_values | integrate_gradients)
        {
#ifdef DEBUG_OUTPUT
          std::cout << "integrate face FEDatas " << fe_number << " " << integrate_values << " "
                    << integrate_gradients << std::endl;
#endif
          fe_data.fe_evaluation_interior->integrate(integrate_values, integrate_gradients);
        }
        if (integrate_values_exterior | integrate_gradients_exterior)
        {
#ifdef DEBUG_OUTPUT
          std::cout << "integrate face exterior FEDatas " << fe_number << " "
                    << integrate_values_exterior << " " << integrate_gradients_exterior
                    << std::endl;
#endif
          fe_data.fe_evaluation_exterior->integrate(integrate_values_exterior,
                                                    integrate_gradients_exterior);
        }
      }
    Base::integrate_face();
  }

  template <unsigned int fe_number_extern>
  void
  set_integration_flags(bool integrate_value, bool integrate_gradient)
  {
    if constexpr(fe_number == fe_number_extern && CFL::Traits::is_fe_data<FEData>::value)
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

  void
  reset_integration_flags_face_and_boundary()
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value)
      {
        integrate_values = false;
        integrate_values_exterior = false;
        integrate_gradients = false;
        integrate_gradients_exterior = false;
      }
    Base::reset_integration_flags_face_and_boundary();
  }

  template <unsigned int fe_number_extern>
  void
  set_integration_flags_face_and_boundary(bool integrate_value, bool integrate_value_exterior,
                                          bool integrate_gradient, bool integrate_gradient_exterior)
  {
    if constexpr(CFL::Traits::is_fe_data_face<FEData>::value && fe_number == fe_number_extern)
      {
        integrate_values |= integrate_value;
        integrate_values_exterior |= integrate_value_exterior;
        integrate_gradients |= integrate_gradient;
        integrate_gradients_exterior |= integrate_gradient_exterior;
#ifdef DEBUG_OUTPUT
        std::cout << "integrate face value: " << fe_number << " " << integrate_values << " "
                  << integrate_value_exterior << std::endl;
        std::cout << "integrate face value exterior: " << fe_number << " "
                  << integrate_values_exterior << " " << integrate_value_exterior << std::endl;
        std::cout << "integrate face gradients: " << fe_number << " " << integrate_gradients << " "
                  << integrate_gradient << std::endl;
        std::cout << "integrate face gradients exterior: " << fe_number << " "
                  << integrate_gradients_exterior << " " << integrate_gradient_exterior
                  << std::endl;
#endif
      }
    else
      Base::template set_integration_flags_face_and_boundary<fe_number_extern>(
        integrate_value, integrate_value_exterior, integrate_gradient, integrate_gradient_exterior);
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
#ifdef DEBUG_OUTPUT
        std::cout << "evaluate cell value: " << fe_number << " " << evaluate_values << " "
                  << evaluate_value << std::endl;
        std::cout << "evaluate cell gradients: " << fe_number << " " << evaluate_gradients << " "
                  << evaluate_gradient << std::endl;
        std::cout << "evaluate cell hessian: " << fe_number << " " << evaluate_hessians << " "
                  << evaluate_hessian << std::endl;
#endif
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
#ifdef DEBUG_OUTPUT
        std::cout << "evaluate face value: " << fe_number << " " << evaluate_values << " "
                  << evaluate_value << std::endl;
        std::cout << "evaluate face gradients: " << fe_number << " " << evaluate_gradients << " "
                  << evaluate_gradient << std::endl;
        std::cout << "evaluate face hessian: " << fe_number << " " << evaluate_hessians << " "
                  << evaluate_hessian << std::endl;
#endif
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
        std::cout << "get face value FEDatas " << fe_number << " " << q << " " << interior << ": ";

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
        std::cout << "submit face value FEDatas " << fe_number << " " << q << std::endl;
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
        std::cout << "submit normal gradient " << fe_number << " " << q << std::endl;
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

  auto
  get_fe_datas() const
  {
    return fe_data;
  }

  template <unsigned int fe_number_extern>
  unsigned int
  dofs_per_cell() const
  {
    if constexpr(fe_number == fe_number_extern)
      return fe_data.fe_evaluation->dofs_per_cell;
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
    if constexpr(fe_number == fe_number_extern)
      return fe_data.fe_evaluation->begin_dof_values();
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

  template <class FEData1, class FEData2, typename... NewTypes,
          typename std::enable_if<
            (CFL::Traits::is_fe_data<FEData1>::value || CFL::Traits::is_fe_data_face<FEData1>::value) &&
            (CFL::Traits::is_fe_data<FEData2>::value || CFL::Traits::is_fe_data_face<FEData2>::value)
            >::type* unused = nullptr>
  auto
  operator,(const FEDatas<FEData1, FEData2, NewTypes...>& new_fe_data) const
  {
    auto a = FEDatas<FEData1, FEData, Types...>(new_fe_data.get_fe_datas(), *this);
    auto b =
      FEDatas<FEData2, NewTypes...>(static_cast<const FEDatas<FEData2, NewTypes...>>(new_fe_data));
    auto c = (a, b);
    return c;
  }

  template <class FEDataOther,
            typename std::enable_if<
            CFL::Traits::is_fe_data<FEDataOther>::value ||
             CFL::Traits::is_fe_data_face<FEDataOther>::value>::type* unused = nullptr>
  auto
  operator,(const FEDatas<FEDataOther>& new_fe_data) const
  {
    return FEDatas<FEDataOther, FEData, Types...>(new_fe_data.get_fe_datas(), *this);
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
  bool integrate_values_exterior = false;
  bool integrate_gradients = false;
  bool integrate_gradients_exterior = false;
  bool evaluate_values = false;
  bool evaluate_gradients = false;
  bool evaluate_hessians = false;
  bool initialized = false;
};

/**
 * Operator overloading function to allow appending FEData to FEDatas
 * \relates FEDatas
 */
template <class FEData, typename... Types>
typename std::enable_if_t<CFL::Traits::is_fe_data<FEData>::value ||
                            CFL::Traits::is_fe_data_face<FEData>::value,
                          FEDatas<FEData, Types...>>
operator,(const FEData& new_fe_data, const FEDatas<Types...>& old_fe_data)
{
  return old_fe_data.operator,(new_fe_data);
}

/**
 * Operator overloading function to allow appending two FEData to form
 * a new container FEDatas
 * \relates FEDatas
 */
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
}

#endif // DEALII_MATRIXFREE_FE_DATA_H
