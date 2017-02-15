#ifndef MATRIX_FREE_INTEGRATOR_H
#define MATRIX_FREE_INTEGRATOR_H

//#include <deal.II/matrix_free/operators.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <cfl/static_for.h>
#include <dealii/operators.h>

using namespace dealii;

template <int dim, typename Number, class FORM, class FEDatas>
class MatrixFreeIntegrator : public ::dealii::MatrixFreeOperators::Base<dim, Number>
{
public:
  // need this to be compatible with the Multigrid interface
  MatrixFreeIntegrator() = default;

  void
  initialize(const MatrixFree<dim, Number>& data_, const std::shared_ptr<FORM>& form_,
             std::shared_ptr<FEDatas> fe_datas_)
  {
    ::dealii::MatrixFreeOperators::Base<dim, Number>::initialize(data_);
    initialize(form_, fe_datas_);
  }

  void
  initialize(const MatrixFree<dim, Number>& data_, const FORM& form_, FEDatas fe_datas_)
  {
    ::dealii::MatrixFreeOperators::Base<dim, Number>::initialize(data_);
    initialize(form_, fe_datas_);
  }

  void
  initialize(const MatrixFree<dim, Number>& data_, const MGConstrainedDoFs& mg_constrained_dofs,
             const unsigned int level, const std::shared_ptr<FORM>& form_,
             std::shared_ptr<FEDatas> fe_datas_)
  {
    ::dealii::MatrixFreeOperators::Base<dim, Number>::initialize(data_, mg_constrained_dofs, level);
    initialize(form_, fe_datas_);
  }

  void
  initialize(const MatrixFree<dim, Number>& data_, const MGConstrainedDoFs& mg_constrained_dofs,
             const unsigned int level, const FORM& form_, FEDatas fe_datas_)
  {
    ::dealii::MatrixFreeOperators::Base<dim, Number>::initialize(data_, mg_constrained_dofs, level);
    initialize(form_, fe_datas_);
  }

  void
  set_nonlinearities(std::vector<bool> nonlinear_components_,
                     LinearAlgebra::distributed::BlockVector<Number> dst)
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
  vmult(LinearAlgebra::distributed::Vector<Number>& dst,
        const LinearAlgebra::distributed::Vector<Number>& src) const
  {
    ::dealii::MatrixFreeOperators::Base<dim, Number>::vmult(dst, src);
  }

  void
  vmult(LinearAlgebra::distributed::BlockVector<Number>& dst,
        const LinearAlgebra::distributed::BlockVector<Number>& src) const
  {
    if (nonlinear_components.size() == 0)
      ::dealii::MatrixFreeOperators::Base<dim, Number>::vmult(dst, src);
    else
    {
      for (unsigned int i = 0; i < dst.n_blocks(); ++i)
      {
        if (!nonlinear_components[i])
          safed_vectors.block(i) = src.block(i);
      }
      ::dealii::MatrixFreeOperators::Base<dim, Number>::vmult(dst, safed_vectors);
      for (unsigned int i = 0; i < dst.n_blocks(); ++i)
      {
        if (nonlinear_components[i])
          dst.block(i) = src.block(i);
      }
    }
  }

  virtual void
  compute_diagonal() override
  {
    if
      constexpr(FEDatas::n == 1)
      {
        Assert((::dealii::MatrixFreeOperators::Base<dim, Number>::data != NULL),
               ExcNotInitialized());
        unsigned int dummy = 0;
        this->inverse_diagonal_entries.reset(
          new DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>());
        LinearAlgebra::distributed::Vector<Number>& inverse_diagonal_vector =
          this->inverse_diagonal_entries->get_vector();
        this->initialize_dof_vector(inverse_diagonal_vector);
        // LinearAlgebra::distributed::Vector<Number> ones;
        // this->initialize_dof_vector(ones);
        // ones = Number(1.);
        // apply_add(inverse_diagonal_vector, ones);

        this->data->cell_loop(
          &MatrixFreeIntegrator::local_diagonal_cell, this, inverse_diagonal_vector, dummy);

        this->set_constrained_entries_to_one(inverse_diagonal_vector);

        const unsigned int local_size = inverse_diagonal_vector.local_size();
        for (unsigned int i = 0; i < local_size; ++i)
          if (std::abs(inverse_diagonal_vector.local_element(i)) >
              std::sqrt(std::numeric_limits<Number>::epsilon()))
            inverse_diagonal_vector.local_element(i) =
              1. / inverse_diagonal_vector.local_element(i);
          else
            inverse_diagonal_vector.local_element(i) = 1.;

        inverse_diagonal_vector.update_ghost_values();
        // inverse_diagonal_vector.print(std::cout);
      }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  // TODO this is hacky and just tries to get comparable output w.r.t. step-37
  void
  local_diagonal_cell(const MatrixFree<dim, Number>& data_,
                      LinearAlgebra::distributed::Vector<Number>& dst, const unsigned int&,
                      const std::pair<unsigned int, unsigned int>& cell_range) const
  {
    (void)data_;
    Assert(&data_ == &(this->get_matrix_free()), ExcInternalError());
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_datas->reinit(cell);
      const /*expr*/ unsigned int tensor_dofs_per_cell =
        fe_datas->template tensor_dofs_per_cell<0>();
      std::vector<VectorizedArray<Number>> local_diagonal_vector(tensor_dofs_per_cell);

      AssertThrow(data_.n_components() == 1, ExcNotImplemented());

      for (unsigned int i = 0; i < fe_datas->template dofs_per_cell<0>(); ++i)
      {
        for (unsigned int j = 0; j < fe_datas->template dofs_per_cell<0>(); ++j)
          fe_datas->template begin_dof_values<0>()[j] = VectorizedArray<Number>();
        fe_datas->template begin_dof_values<0>()[i] = 1.;
        do_operation_on_cell(*fe_datas, cell);
        local_diagonal_vector[i] = fe_datas->template begin_dof_values<0>()[i];
      }
      for (unsigned int i = 0; i < fe_datas->template tensor_dofs_per_cell<0>(); ++i)
        fe_datas->template begin_dof_values<0>()[i] = local_diagonal_vector[i];
      fe_datas->distribute_local_to_global(dst);
    }
  }

private:
  std::shared_ptr<const FORM> form = nullptr;
  std::shared_ptr<FEDatas> fe_datas = nullptr;
  bool use_cell = false;
  bool use_face = false;
  bool use_boundary = false;
  std::vector<bool> nonlinear_components;
  mutable dealii::LinearAlgebra::distributed::BlockVector<double> safed_vectors;

  // convenience function to avoid shared_ptr
  void
  initialize(const FORM& form_, FEDatas& fe_datas_)
  {
    initialize(std::make_shared<FORM>(form_), std::make_shared<FEDatas>(fe_datas_));
  }

  void
  initialize(const std::shared_ptr<FORM>& form_, std::shared_ptr<FEDatas>& fe_datas_)
  {
    form = form_;
    fe_datas = fe_datas_;
    // TODO: Determine from form.
    use_cell = true;
    use_face = false;
    use_boundary = false;
    form->set_evaluation_flags(*fe_datas);
    form->set_integration_flags(*fe_datas);
    Assert(this->data != nullptr, ExcNotInitialized());
    fe_datas->initialize(*(this->data));
  }

  virtual void
  apply_add(LinearAlgebra::distributed::Vector<Number>& dst,
            const LinearAlgebra::distributed::Vector<Number>& src) const override
  {
    if (use_cell)
      dealii::MatrixFreeOperators::Base<dim, Number>::data->cell_loop(
        &MatrixFreeIntegrator::local_apply_cell, this, dst, src);
    if (use_face)
    {
      /* Base<dim, Number>::data->face_loop (&MatrixFreeIntegrator::local_apply_face,
                                             this, dst, src);*/
    }
    if (use_boundary)
    {
      /* Base<dim, Number>::data->boundary_loop (&MatrixFreeIntegrator::local_apply_boundary,
                                                 this, dst, src);*/
    }
  }

  virtual void
  apply_add(LinearAlgebra::distributed::BlockVector<Number>& dst,
            const LinearAlgebra::distributed::BlockVector<Number>& src) const override
  {
    if (use_cell)
      dealii::MatrixFreeOperators::Base<dim, Number>::data->cell_loop(
        &MatrixFreeIntegrator::local_apply_cell, this, dst, src);
    if (use_face)
    {
      /* Base<dim, Number>::data->face_loop (&MatrixFreeIntegrator::local_apply_face,
                                             this, dst, src);*/
    }
    if (use_boundary)
    {
      /* Base<dim, Number>::data->boundary_loop (&MatrixFreeIntegrator::local_apply_boundary,
                                                 this, dst, src);*/
    }
  }

  template <class FEEvaluation>
  void
  do_operation_on_cell(FEEvaluation& phi, const unsigned int /*cell*/) const
  {
    phi.evaluate();
    constexpr unsigned int n_q_points = phi.get_n_q_points();
    // static_for_old<0, n_q_points>()([&](int q)
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      form->evaluate(phi, q);
    }

    phi.integrate();
  }

  template <class VectorType>
  void
  local_apply_cell(const MatrixFree<dim, Number>& data_, VectorType& dst, const VectorType& src,
                   const std::pair<unsigned int, unsigned int>& cell_range) const
  {
    static_assert(std::is_same_v<VectorType, LinearAlgebra::distributed::Vector<Number>> ||
                    std::is_same_v<VectorType, LinearAlgebra::distributed::BlockVector<Number>>,
                  "This is only implemented for LinearAlgebra::distributed::Vector<Number> "
                  "and LinearAlgebra::distributed::BlockVector<Number> objects!");
    (void)data_;
    Assert(&data_ == &(this->get_matrix_free()), ExcInternalError());
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_datas->reinit(cell);
      fe_datas->read_dof_values(src);
      do_operation_on_cell(*fe_datas, cell);
      fe_datas->distribute_local_to_global(dst);
    }
  }
};

#endif // MATRIX_FREE_INTEGRATOR_H
