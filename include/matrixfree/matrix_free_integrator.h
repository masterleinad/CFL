#ifndef MATRIX_FREE_INTEGRATOR_H
#define MATRIX_FREE_INTEGRATOR_H

#include <deal.II/matrix_free/operators.h>

#include <cfl/fefunctions.h> //for BlockVectors
#include <cfl/traits.h>
#include <deal.II/lac/la_parallel_block_vector.h>

template <int dim, typename VectorType, class Enable = void>
class MatrixFreeIntegratorBaseBase;

template <int dim, typename VectorType>
class MatrixFreeIntegratorBaseBase<
  dim, VectorType, typename std::enable_if_t<!CFL::Traits::is_block_vector<VectorType>::value>>
  : public dealii::MatrixFreeOperators::Base<dim, VectorType>
{
public:
  using Base = dealii::MatrixFreeOperators::Base<dim, VectorType>;
};

template <int dim, typename VectorType>
class MatrixFreeIntegratorBaseBase<
  dim, VectorType, typename std::enable_if_t<CFL::Traits::is_block_vector<VectorType>::value>>
  : public dealii::MatrixFreeOperators::Base<dim, VectorType>
{
public:
  using Base = dealii::MatrixFreeOperators::Base<dim, VectorType>;
};

template <int dim, typename VectorType, class FORM, class FEDatas>
class MatrixFreeIntegratorBase : public MatrixFreeIntegratorBaseBase<dim, VectorType>
{
public:
  using Number = typename VectorType::value_type;
  using Base = MatrixFreeIntegratorBaseBase<dim, VectorType>;

  void
  initialize(const std::shared_ptr<const dealii::MatrixFree<dim, Number>>& data_,
             const std::shared_ptr<FORM>& form_, std::shared_ptr<FEDatas> fe_datas_)
  {
    Base::Base::initialize(data_);
    initialize(form_, fe_datas_);
  }

  void
  initialize(const dealii::MatrixFree<dim, Number>& data_, const FORM& form_, FEDatas fe_datas_)
  {
    Base::Base::initialize(std::make_shared<const dealii::MatrixFree<dim, Number>>(data_));
    initialize(form_, fe_datas_);
  }

protected:
  std::shared_ptr<const FORM> form = nullptr;
  std::shared_ptr<FEDatas> fe_datas = nullptr;
  std::array<bool, 3> use_objects{ { /*cell*/ false, /*face*/ false, /*boundary*/ false } };

  // convenience function to avoid shared_ptr
  void
  initialize(const FORM& form_, FEDatas& fe_datas_)
  {
    initialize(std::make_shared<FORM>(form_), std::make_shared<FEDatas>(fe_datas_));
  }

  void
  initialize(const std::shared_ptr<FORM>& form_, std::shared_ptr<FEDatas>& fe_datas_)
  {
    // first retrieve information about integration loops to use from the Forms object
    FORM::get_form_kinds(use_objects);
    const bool use_cell = use_objects[0];
    const bool use_face = use_objects[1];
    const bool use_boundary = use_objects[2];

    // then initialize the objects given appropriately
    form = form_;
    fe_datas = fe_datas_;

    // we don't need to share these so initailize them already here.
    if (use_cell)
    {
      form->set_evaluation_flags(*fe_datas);
      form->set_integration_flags(*fe_datas);
    }

    if (use_face || use_boundary)
      form->set_evaluation_flags_face(*fe_datas);

    Assert(this->data != nullptr, dealii::ExcNotInitialized());
    fe_datas->initialize(*(this->data));
  }

  void
  apply_add(VectorType& dst, const VectorType& src) const override
  {
    static_assert(
      std::is_same<VectorType, dealii::LinearAlgebra::distributed::Vector<Number>>::value ||
        std::is_same<VectorType, dealii::LinearAlgebra::distributed::BlockVector<Number>>::value,
      "This is only implemented for dealii::LinearAlgebra::distributed::Vector<Number> "
      "and dealii::LinearAlgebra::distributed::BlockVector<Number> objects!");
    const bool use_cell = use_objects[0];
    const bool use_face = use_objects[1];
    const bool use_boundary = use_objects[2];

    if (use_cell | use_face | use_boundary)
    {
      Base::data->loop(use_cell ? &MatrixFreeIntegratorBase::local_apply : nullptr,
                       use_face ? &MatrixFreeIntegratorBase::local_apply_face : nullptr,
                       use_boundary ? &MatrixFreeIntegratorBase::local_apply_boundary : nullptr,
                       this,
                       dst,
                       src);
    }
  }

  template <class FEEvaluation>
  void
  do_operation_on_cell(FEEvaluation& phi, const unsigned int /*cell*/) const
  {
    phi.evaluate();
    constexpr unsigned int n_q_points = FEEvaluation::get_n_q_points();
    // static_for_old<0, n_q_points>()([&](int q)
    for (unsigned int q = 0; q < n_q_points; ++q)
      form->evaluate(phi, q);

    phi.integrate();
  }

  template <class FEEvaluation>
  void
  do_operation_on_face(FEEvaluation& phi, const unsigned int /*cell*/) const
  {
    phi.evaluate_face();
    constexpr unsigned int n_q_points = FEEvaluation::get_n_q_points_face();
    // static_for_old<0, n_q_points>()([&](int q)
    for (unsigned int q = 0; q < n_q_points; ++q)
      form->evaluate_face(phi, q);

    phi.integrate_face();
  }

  template <class FEEvaluation>
  void
  do_operation_on_boundary(FEEvaluation& phi, const unsigned int /*cell*/) const
  {
    phi.template evaluate_face<true, false>();
    constexpr unsigned int n_q_points = FEEvaluation::get_n_q_points_face();
    // static_for_old<0, n_q_points>()([&](int q)
    for (unsigned int q = 0; q < n_q_points; ++q)
      form->evaluate_boundary(phi, q);

    phi.integrate_face();
  }

  void local_apply([[maybe_unused]] const dealii::MatrixFree<dim, Number>& data_, VectorType& dst,
                   const VectorType& src,
                   const std::pair<unsigned int, unsigned int>& cell_range) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Begin cell loop" << std::endl;
#endif
    Assert(&data_ == (this->get_matrix_free()).get(), dealii::ExcInternalError());
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_datas->reinit(cell);
      fe_datas->read_dof_values(src);
      do_operation_on_cell(*fe_datas, cell);
      fe_datas->distribute_local_to_global(dst);
    }
#ifdef DEBUG_OUTPUT
    std::cout << "End cell loop" << std::endl;
#endif
  }

  void local_apply_face([[maybe_unused]] const dealii::MatrixFree<dim, Number>& data_,
                        VectorType& dst, const VectorType& src,
                        const std::pair<unsigned int, unsigned int>& face_range) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Begin face loop" << std::endl;
#endif
    Assert(&data_ == (this->get_matrix_free()).get(), dealii::ExcInternalError());
    fe_datas->reset_integration_flags_face_and_boundary();
    form->set_integration_flags_face(*fe_datas);
    for (unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_datas->reinit_face(face);
      fe_datas->read_dof_values_face(src);
      do_operation_on_face(*fe_datas, face);
      fe_datas->distribute_local_to_global_face(dst);
    }
#ifdef DEBUG_OUTPUT
    std::cout << "End face loop" << std::endl;
#endif
  }

  void local_apply_boundary([[maybe_unused]] const dealii::MatrixFree<dim, Number>& data_,
                            VectorType& dst, const VectorType& src,
                            const std::pair<unsigned int, unsigned int>& face_range) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Begin boundary loop" << std::endl;
#endif
    Assert(&data_ == (this->get_matrix_free()).get(), dealii::ExcInternalError());
    fe_datas->reset_integration_flags_face_and_boundary();
    form->set_integration_flags_boundary(*fe_datas);
    for (unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_datas->reinit_boundary(face);
      // We never need values from the "neighboring" face as there is none.
      fe_datas->template read_dof_values_face<VectorType, true, false>(src);
      do_operation_on_boundary(*fe_datas, face);
      fe_datas->template distribute_local_to_global_face<VectorType, true, false>(dst);
    }
#ifdef DEBUG_OUTPUT
    std::cout << "End boundary loop" << std::endl;
#endif
  }
};

template <int dim, typename VectorType, class FORM, class FEDatas, class Enable = void>
class MatrixFreeIntegrator;

template <int dim, typename VectorType, class FORM, class FEDatas>
class MatrixFreeIntegrator<
  dim, VectorType, FORM, FEDatas,
  typename std::enable_if_t<!CFL::Traits::is_block_vector<VectorType>::value>>
  : public MatrixFreeIntegratorBase<dim, VectorType, FORM, FEDatas>
{
public:
  using Number = typename VectorType::value_type;
  using Base = MatrixFreeIntegratorBase<dim, VectorType, FORM, FEDatas>;
  using Base::initialize;

  void
  initialize(const std::shared_ptr<const dealii::MatrixFree<dim, Number>>& data_,
             const dealii::MGConstrainedDoFs& mg_constrained_dofs, const unsigned int level,
             const std::shared_ptr<FORM>& form_, std::shared_ptr<FEDatas> fe_datas_)
  {
    Base::Base::Base::initialize(data_, mg_constrained_dofs, level);
    initialize(form_, fe_datas_);
  }

  void
  initialize(const dealii::MatrixFree<dim, Number>& data_,
             const dealii::MGConstrainedDoFs& mg_constrained_dofs, const unsigned int level,
             const FORM& form_, FEDatas fe_datas_)
  {
    Base::Base::Base::initialize(
      std::make_shared<const dealii::MatrixFree<dim, Number>>(data_), mg_constrained_dofs, level);
    initialize(form_, fe_datas_);
  }

  void
  vmult(VectorType& dst, const VectorType& src) const
  {
    Base::vmult(dst, src);
  }

  void
  compute_diagonal() override
  {
    static_assert(FEDatas::n == 1, "This is for one compoennt!");
    Assert((Base::data != nullptr), dealii::ExcNotInitialized());
    unsigned int dummy = 0;
    this->inverse_diagonal_entries.reset(new dealii::DiagonalMatrix<VectorType>());
    VectorType& inverse_diagonal_vector = this->inverse_diagonal_entries->get_vector();
    this->initialize_dof_vector(inverse_diagonal_vector);
    // VectorType ones;
    // this->initialize_dof_vector(ones);
    // ones = Number(1.);
    // apply_add(inverse_diagonal_vector, ones);

    this->data->cell_loop(
      &MatrixFreeIntegrator::local_diagonal_cell, this, inverse_diagonal_vector, dummy);

    this->set_constrained_entries_to_one(inverse_diagonal_vector);

    const unsigned int local_size = inverse_diagonal_vector.local_size();
    for (unsigned int i = 0; i < local_size; ++i)
    {
      if (std::abs(inverse_diagonal_vector.local_element(i)) >
          std::sqrt(std::numeric_limits<Number>::epsilon()))
      {
        inverse_diagonal_vector.local_element(i) = 1. / inverse_diagonal_vector.local_element(i);
      }
      else
        inverse_diagonal_vector.local_element(i) = 1.;
    }

    inverse_diagonal_vector.update_ghost_values();
    // inverse_diagonal_vector.print(std::cout);
  }

  // TODO(darndt): this is hacky and just tries to get comparable output w.r.t. step-37
  void local_diagonal_cell([[maybe_unused]] const dealii::MatrixFree<dim, Number>& data_,
                           VectorType& dst, const unsigned int& /*unused*/,
                           const std::pair<unsigned int, unsigned int>& cell_range) const
  {
    Assert(&data_ == (this->get_matrix_free()).get(), dealii::ExcInternalError());
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      Base::fe_datas->reinit(cell);
      const /*expr*/ unsigned int tensor_dofs_per_cell =
        Base::fe_datas->template tensor_dofs_per_cell<0>();
      std::vector<dealii::VectorizedArray<Number>> local_diagonal_vector(tensor_dofs_per_cell);

      AssertThrow(data_.n_components() == 1, dealii::ExcNotImplemented());

      for (unsigned int i = 0; i < Base::fe_datas->template dofs_per_cell<0>(); ++i)
      {
        for (unsigned int j = 0; j < Base::fe_datas->template dofs_per_cell<0>(); ++j)
          Base::fe_datas->template begin_dof_values<0>()[j] = dealii::VectorizedArray<Number>();
        Base::fe_datas->template begin_dof_values<0>()[i] = 1.;
        Base::do_operation_on_cell(*Base::fe_datas, cell);
        local_diagonal_vector[i] = Base::fe_datas->template begin_dof_values<0>()[i];
      }
      for (unsigned int i = 0; i < Base::fe_datas->template tensor_dofs_per_cell<0>(); ++i)
        Base::fe_datas->template begin_dof_values<0>()[i] = local_diagonal_vector[i];
      Base::fe_datas->distribute_local_to_global(dst);
    }
  }
};

template <int dim, typename VectorType, class FORM, class FEDatas>
class MatrixFreeIntegrator<
  dim, VectorType, FORM, FEDatas,
  typename std::enable_if_t<CFL::Traits::is_block_vector<VectorType>::value>>
  : public MatrixFreeIntegratorBase<dim, VectorType, FORM, FEDatas>
{
public:
  using Number = typename VectorType::value_type;
  using Base = MatrixFreeIntegratorBase<dim, VectorType, FORM, FEDatas>;
  using Base::initialize;

  void
  initialize(const std::shared_ptr<const dealii::MatrixFree<dim, Number>>& data_,
             const std::vector<dealii::MGConstrainedDoFs>& mg_constrained_dofs,
             const unsigned int level, const std::shared_ptr<FORM>& form_,
             std::shared_ptr<FEDatas> fe_datas_)
  {
    Base::Base::Base::initialize(data_, mg_constrained_dofs, level);
    initialize(form_, fe_datas_);
  }

  void
  initialize(const dealii::MatrixFree<dim, Number>& data,
             const std::vector<dealii::MGConstrainedDoFs>& mg_constrained_dofs,
             const unsigned int level, const FORM& form_, FEDatas fe_datas_)
  {
    Base::Base::Base::initialize(
      std::make_shared<const dealii::MatrixFree<dim, Number>>(data), mg_constrained_dofs, level);
    initialize(form_, fe_datas_);
  }

  void
  set_nonlinearities(std::vector<bool> nonlinear_components_,
                     dealii::LinearAlgebra::distributed::BlockVector<Number> dst)
  {
    nonlinear_components = nonlinear_components_;
    safed_vectors.reinit(dst);
    for (unsigned int i = 0; i < dst.n_blocks(); ++i)
    {
      AssertIndexRange(i, dst.n_blocks());
      if (nonlinear_components[i])
        safed_vectors.block(i) = dst.block(i);
    }
  }

  void
  vmult(dealii::LinearAlgebra::distributed::BlockVector<Number>& dst,
        const dealii::LinearAlgebra::distributed::BlockVector<Number>& src) const
  {
    if (nonlinear_components.empty())
      Base::vmult(dst, src);
    else
    {
      for (unsigned int i = 0; i < dst.n_blocks(); ++i)
      {
        if (!nonlinear_components[i])
          safed_vectors.block(i) = src.block(i);
      }
      Base::vmult(dst, safed_vectors);
      for (unsigned int i = 0; i < dst.n_blocks(); ++i)
      {
        if (nonlinear_components[i])
          dst.block(i) = src.block(i);
      }
    }
  }

  void
  compute_diagonal() override
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }

private:
  std::vector<bool> nonlinear_components;
  mutable VectorType safed_vectors;
};

#endif // MATRIX_FREE_INTEGRATOR_H
