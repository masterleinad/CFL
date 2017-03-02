// ---------------------------------------------------------------------
//
// Copyright (C) 2011 - 2016 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii__matrix_free_operators_new_h
#define dealii__matrix_free_operators_new_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector_view.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

DEAL_II_NAMESPACE_OPEN

namespace MatrixFreeOperators
{
/**
 * Abstract base class for matrix-free operators which can be used both at
 * the finest mesh or at a certain level in geometric multigrid.
 *
 * A derived class has to implement apply_add() method as well as
 * compute_diagonal() to initialize the protected member inverse_diagonal_entries.
 * In case of a non-symmetric operator, Tapply_add() should be additionally
 * implemented.
 *
 * @author Denis Davydov, 2016
 */
template <int dim, typename VectorType = LinearAlgebra::distributed::BlockVector<double>>
class BlockBase : public Subscriptor
{
public:
  /**
   * value_type typedef.
   */
  using value_type = typename VectorType::value_type;

  /**
   * size_type needed for preconditioner classes.
   */
  using size_type = typename VectorType::size_type;

  /**
   * Default constructor.
   */
  BlockBase();

  /**
   * Virtual destructor.
   */
  virtual ~BlockBase();

  /**
   * Release all memory and return to a state just like after having called
   * the default constructor.
   */
  virtual void clear();

  /**
   * Initialize operator on fine scale.
   */
  void initialize(std_cxx11::shared_ptr<const MatrixFree<dim, value_type>> data_);

  /**
   * Initialize operator on a level @p level for multiple FiniteElements.
   */
  void initialize(std_cxx11::shared_ptr<const MatrixFree<dim, value_type>> data_,
                  const std::vector<MGConstrainedDoFs>& mg_constrained_dofs,
                  const unsigned int level);

  /**
   * Return the dimension of the codomain (or range) space.
   */
  size_type m() const;

  /**
   * Return the dimension of the domain space.
   */
  size_type n() const;

  /**
   * vmult operator for interface for multiple FiniteElements.
   */
  void vmult_interface_down(VectorType& dst, const VectorType& src) const;

  /**
   * vmult operator for interface for multiple FiniteElements.
   */
  void vmult_interface_up(VectorType& dst, const VectorType& src) const;

  /**
   * Matrix-vector multiplication for multiple FiniteElements.
   */
  void vmult(VectorType& dst, const VectorType& src) const;

  /**
   * Transpose matrix-vector multiplication for multiple FiniteElements.
   */
  void Tvmult(VectorType& dst, const VectorType& src) const;

  /**
   * Adding Matrix-vector multiplication for multiple FiniteElements.
   */
  void vmult_add(VectorType& dst, const VectorType& src) const;

  /**
   * Adding transpose matrix-vector multiplication for multiple FiniteElements.
   */
  void Tvmult_add(VectorType& dst, const VectorType& src) const;

  /**
   * Returns the value of the matrix entry (row,col). In matrix-free context
   * this function is valid only for row==col when diagonal is initialized.
   */
  value_type el(const unsigned int row, const unsigned int col) const;

  /**
   * Determine an estimate for the memory consumption (in bytes) of this object.
   */
  virtual std::size_t memory_consumption() const;

  /**
   * A wrapper for initialize_dof_vector() of MatrixFree object.
   */
  void initialize_dof_vector(VectorType& vec) const;

  /**
   * Compute diagonal of this operator.
   *
   * A derived class needs to implement this function and resize and fill
   * the protected member inverse_diagonal_entries accordingly.
   */
  virtual void compute_diagonal() = 0;

  /**
   * Get read access to the MatrixFree object stored with this operator.
   */
  std_cxx11::shared_ptr<const MatrixFree<dim, value_type>> get_matrix_free() const;

  /**
   * Get read access to the inverse diagonal of this operator.
   */
  const std_cxx11::shared_ptr<DiagonalMatrix<VectorType>>& get_matrix_diagonal_inverse() const;

  /**
   * Apply the Jacobi preconditioner, which multiplies every element of the
   * <tt>src</tt> vector by the inverse of the respective diagonal element and
   * multiplies the result with the relaxation factor <tt>omega</tt>.
   */
  void precondition_Jacobi(VectorType& dst, const VectorType& src, const value_type omega) const;

protected:
  /**
   * Set constrained entries (both from hanging nodes and edge constraints)
   * of @p dst to one.
   */
  void set_constrained_entries_to_one(VectorType& dst) const;

  /**
   * Apply operator to @p src and add result in @p dst.
   */
  virtual void apply_add(VectorType& dst, const VectorType& src) const = 0;

  /**
   * Apply transpose operator to @p src and add result in @p dst.
   *
   * Default implementation is to call apply_add().
   */
  virtual void Tapply_add(VectorType& dst, const VectorType& src) const;

  /**
   * MatrixFree object to be used with this operator.
   */
  std_cxx11::shared_ptr<const MatrixFree<dim, value_type>> data;

  /**
   * A shared pointer to a diagonal matrix that stores the inverse of
   * diagonal elements as a vector.
   */
  std_cxx11::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal_entries;

private:
  /**
   * Indices of DoFs on edge in case the operator is used in GMG context.
   */
  std::vector<std::vector<unsigned int>> edge_constrained_indices;

  /**
   * Auxiliary vector.
   */
  mutable std::vector<std::vector<std::pair<value_type, value_type>>> edge_constrained_values;

  /**
   * A flag which determines whether or not this operator has interface
   * matrices in GMG context.
   */
  bool have_interface_matrices{ false };

  /**
   * Function which implements vmult_add (@p transpose = false) and
   * Tvmult_add (@p transpose = true).
   */
  void mult_add(VectorType& dst, const VectorType& src, const bool transpose) const;

  /**
   * Adjust the ghost range of the vectors to the storage requirements of
   * the underlying MatrixFree class. This is used inside the mult_add() as
   * well as vmult_interface_up() and vmult_interface_down() methods in
   * order to ensure that the cell loops will be able to access the ghost
   * indices with the correct local indices.
   */
  void adjust_ghost_range_if_necessary(const VectorType& src) const;
};

//----------------- BlockBase operator -----------------------------
template <int dim, typename VectorType>
BlockBase<dim, VectorType>::~BlockBase() = default;

template <int dim, typename VectorType>
BlockBase<dim, VectorType>::BlockBase()
  : Subscriptor()
  , data(nullptr)
{
}

template <int dim, typename VectorType>
typename BlockBase<dim, VectorType>::size_type
BlockBase<dim, VectorType>::m() const
{
  Assert(data != NULL, ExcNotInitialized());
  return data->get_vector_partitioner()->size();
}

template <int dim, typename VectorType>
typename BlockBase<dim, VectorType>::size_type
BlockBase<dim, VectorType>::n() const
{
  return m();
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::clear()
{
  data = nullptr;
  inverse_diagonal_entries.reset();
}

template <int dim, typename VectorType>
typename BlockBase<dim, VectorType>::value_type
BlockBase<dim, VectorType>::el(const unsigned int row,
                               [[maybe_unused]] const unsigned int col) const
{
  Assert(row == col, ExcNotImplemented());
  Assert(inverse_diagonal_entries.get() != NULL && inverse_diagonal_entries->m() > 0,
         ExcNotInitialized());
  return 1.0 / (*inverse_diagonal_entries)(row, row);
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::initialize_dof_vector(VectorType& vec) const
{
  Assert(data != NULL, ExcNotInitialized());
  for (unsigned int i = 0; i < vec.n_blocks(); ++i)
  {
    if (!vec.block(i).partitioners_are_compatible(*data->get_dof_info(i).vector_partitioner))
      data->initialize_dof_vector(vec.block(i));
    Assert(
      vec.block(i).partitioners_are_globally_compatible(*data->get_dof_info(0).vector_partitioner),
      ExcInternalError());
  }
  vec.collect_sizes();
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::initialize(
  std_cxx11::shared_ptr<const MatrixFree<dim, value_type>> data_)
{
  data = data_;
  edge_constrained_indices.resize(data->n_components());
  edge_constrained_indices[0].clear();
  edge_constrained_values.resize(data->n_components());
  have_interface_matrices = false;
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::initialize(
  std_cxx11::shared_ptr<const MatrixFree<dim, value_type>> data_,
  const std::vector<MGConstrainedDoFs>& mg_constrained_dofs, const unsigned int level)
{
  AssertThrow(level != numbers::invalid_unsigned_int, ExcMessage("level is not set"));
  AssertDimension(mg_constrained_dofs.size(), data_->n_components());
  edge_constrained_indices.resize(data_->n_components());
  edge_constrained_values.resize(data_->n_components());
  if (data_->n_macro_cells() > 0)
    AssertDimension(static_cast<int>(level), data_->get_cell_iterator(0, 0)->level());

  data = data_;

  for (unsigned int i = 0; i < data->n_components(); ++i)
  {
    // setup edge_constrained indices
    const auto i_size = static_cast<size_t>(i);
    std::vector<types::global_dof_index> interface_indices;
    mg_constrained_dofs[i_size].get_refinement_edge_indices(level).fill_index_vector(
      interface_indices);
    edge_constrained_indices[i_size].clear();
    edge_constrained_indices[i_size].reserve(interface_indices.size());
    edge_constrained_values[i_size].resize(interface_indices.size());
    const IndexSet& locally_owned = data->get_dof_handler(i).locally_owned_mg_dofs(level);
    for (const size_t index : interface_indices)
    {
      if (locally_owned.is_element(index))
        edge_constrained_indices[i_size].push_back(locally_owned.index_within_set(index));
    }
    have_interface_matrices |=
      Utilities::MPI::max(static_cast<unsigned int>(edge_constrained_indices[i_size].size()),
                          data_->get_vector_partitioner()->get_mpi_communicator()) > 0;
  }
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::set_constrained_entries_to_one(VectorType& dst) const
{
  for (unsigned int j = 0; j < dst.n_blocks; ++j)
  {
    const std::vector<unsigned int>& constrained_dofs = data->get_constrained_dofs(j);
    for (unsigned int constrained_dof : constrained_dofs)
      dst.block(j).local_element(constrained_dof) = 1.;
    for (unsigned int i = 0; i < edge_constrained_indices[j].size(); ++i)
      dst.block(j).local_element(edge_constrained_indices[j][i]) = 1.;
  }
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::vmult(VectorType& dst, const VectorType& src) const
{
  dst = 0.;
  vmult_add(dst, src);
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::vmult_add(VectorType& dst, const VectorType& src) const
{
  mult_add(dst, src, false);
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::Tvmult_add(VectorType& dst, const VectorType& src) const
{
  mult_add(dst, src, true);
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::mult_add(VectorType& dst, const VectorType& src,
                                     const bool transpose) const
{
  AssertDimension(dst.size(), src.size());
  AssertDimension(dst.n_blocks(), src.n_blocks());
  // AssertDimension(edge_constrained_values.size(), dst.n_blocks());
  adjust_ghost_range_if_necessary(src);
  adjust_ghost_range_if_necessary(dst);

  // set zero Dirichlet values on the input vector (and remember the src and
  // dst values because we need to reset them at the end)
  for (size_t i = 0; i < dst.n_blocks(); ++i)
  {
    for (size_t j = 0; j < edge_constrained_indices[i].size(); ++j)
    {
      edge_constrained_values[i][j] = std::pair<value_type, value_type>(
        src.block(i).local_element(edge_constrained_indices[i][j]),
        dst.block(i).local_element(edge_constrained_indices[i][j]));
      const_cast<VectorType&>(src).block(i).local_element(edge_constrained_indices[i][j]) = 0.;
    }
  }

  if (transpose)
    Tapply_add(dst, src);
  else
    apply_add(dst, src);

  for (unsigned int i = 0; static_cast<size_t>(i) < dst.n_blocks(); ++i)
  {
    const auto i_size = static_cast<size_t>(i);
    const std::vector<unsigned int>& constrained_dofs = data->get_constrained_dofs(i);
    for (unsigned int constrained_dof : constrained_dofs)
    {
      dst.block(i_size).local_element(constrained_dof) +=
        src.block(i_size).local_element(constrained_dof);
    }
  }

  // reset edge constrained values, multiply by unit matrix and add into
  // destination
  for (size_t i = 0; i < dst.n_blocks(); ++i)
  {
    for (size_t j = 0; j < edge_constrained_indices[i].size(); ++j)
    {
      const_cast<VectorType&>(src).block(i).local_element(edge_constrained_indices[i][j]) =
        edge_constrained_values[i][j].first;
      dst.block(i).local_element(edge_constrained_indices[i][j]) =
        edge_constrained_values[i][j].second + edge_constrained_values[i][j].first;
    }
  }
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::adjust_ghost_range_if_necessary(const VectorType& src) const
{
  for (unsigned int i = 0; static_cast<size_t>(i) < src.n_blocks(); ++i)
  {
    auto i_size = static_cast<size_t>(i);
    const auto dof_info = data->get_dof_info(i);
    // If both vectors use the same partitioner -> done
    const auto& src_vector = src.block(i_size);
    if (src_vector.get_partitioner().get() == dof_info.vector_partitioner.get())
      return;

    // If not, assert that the local ranges are the same and reset to the
    // current partitioner
    Assert(src_vector.get_partitioner()->local_size() == dof_info.vector_partitioner->local_size(),
           ExcMessage("The vector passed to the vmult() function does not have "
                      "the correct size for compatibility with MatrixFree."));

    // copy the vector content to a temporary vector so that it does not get
    // lost
    VectorView<value_type> view_src_in(src_vector.local_size(), src_vector.begin());
    const Vector<value_type>& copy_vec = view_src_in;
    const_cast<VectorType&>(src).block(i_size).reinit(dof_info.vector_partitioner);
    VectorView<value_type> view_src_out(src_vector.local_size(), src_vector.begin());
    static_cast<Vector<value_type>&>(view_src_out) = copy_vec;
  }
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::vmult_interface_down(VectorType& dst, const VectorType& src) const
{
  AssertDimension(dst.size(), src.size());
  adjust_ghost_range_if_necessary(src);
  adjust_ghost_range_if_necessary(dst);

  for (size_t i = 0; i < dst.size(); ++i)
    dst.block(i) = 0;

  if (!have_interface_matrices)
    return;

  // set zero Dirichlet values on the input vector (and remember the src and
  // dst values because we need to reset them at the end)
  for (size_t i = 0; i < dst.size(); ++i)
  {
    for (size_t j = 0; j < edge_constrained_indices.size(); ++j)
    {
      edge_constrained_values[i][j] = std::pair<value_type, value_type>(
        src.block(i).local_element(edge_constrained_indices[i][j]),
        dst.block(i).local_element(edge_constrained_indices[i][j]));
      const_cast<VectorType&>(src.block(i)).local_element(edge_constrained_indices[i][j]) = 0.;
    }
  }

  apply_add(dst, src);

  for (unsigned int i = 0; static_cast<size_t>(i) < dst.size(); ++i)
  {
    const auto i_size = static_cast<size_t>(i);
    for (unsigned int c = 0, j = 0; static_cast<size_t>(j) < edge_constrained_indices.size(); ++j)
    {
      const auto j_size = static_cast<size_t>(j);
      for (; c < edge_constrained_indices[i_size][j_size]; ++c)
        dst.block(i_size).local_element(c) = 0.;
      ++c;

      // reset the src values
      const_cast<VectorType&>(src.block(i_size))
        .local_element(edge_constrained_indices[i_size][j_size]) =
        edge_constrained_values[i_size][j_size].first;

      for (; static_cast<size_t>(c) < dst.local_size(); ++c)
        dst.block(i_size).local_element(c) = 0.;
    }
  }
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::vmult_interface_up(VectorType& dst, const VectorType& src) const
{
  AssertDimension(dst.size(), src.size());
  adjust_ghost_range_if_necessary(src);
  adjust_ghost_range_if_necessary(dst);

  dst = 0;

  if (!have_interface_matrices)
    return;

  VectorType src_cpy = src;

  for (unsigned int i = 0; static_cast<size_t>(i) < dst.size(); ++i)
  {
    const auto i_size = static_cast<size_t>(i);
    const VectorType& src_vector = src_cpy.block(i_size);
    unsigned int c = 0;

    for (unsigned int j = 0; static_cast<size_t>(j) < edge_constrained_indices[i_size].size(); ++j)
    {
      const auto j_size = static_cast<size_t>(j);
      for (; c < edge_constrained_indices[i_size][j_size]; ++c)
        src_vector.local_element(c) = 0.;
      ++c;
    }
    for (; c < src_vector.local_size(); ++c)
      src_vector.local_element(c) = 0.;
  }

  apply_add(dst, src_cpy);

  for (size_t i = 0; i < dst.size(); ++i)
  {
    for (size_t j = 0; j < edge_constrained_indices[i].size(); ++j)
      dst.block(i).local_element(edge_constrained_indices[i][j]) = 0.;
  }
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::Tvmult(VectorType& dst, const VectorType& src) const
{
  dst = 0;
  Tvmult_add(dst, src);
}

template <int dim, typename VectorType>
std::size_t
BlockBase<dim, VectorType>::memory_consumption() const
{
  return inverse_diagonal_entries != nullptr ? inverse_diagonal_entries->memory_consumption()
                                             : sizeof(*this);
}

template <int dim, typename VectorType>
std_cxx11::shared_ptr<const MatrixFree<dim, typename BlockBase<dim, VectorType>::value_type>>
BlockBase<dim, VectorType>::get_matrix_free() const
{
  return data;
}

template <int dim, typename VectorType>
const std_cxx11::shared_ptr<DiagonalMatrix<VectorType>>&
BlockBase<dim, VectorType>::get_matrix_diagonal_inverse() const
{
  Assert(inverse_diagonal_entries.get() != NULL && inverse_diagonal_entries->m() > 0,
         ExcNotInitialized());
  return inverse_diagonal_entries;
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::Tapply_add(VectorType& dst, const VectorType& src) const
{
  apply_add(dst, src);
}

template <int dim, typename VectorType>
void
BlockBase<dim, VectorType>::precondition_Jacobi(VectorType& dst, const VectorType& src,
                                                const value_type omega) const
{
  Assert(inverse_diagonal_entries.get() && inverse_diagonal_entries->m() > 0, ExcNotInitialized());
  inverse_diagonal_entries->vmult(dst, src);
  dst *= omega;
}
} // end of namespace MatrixFreeOperators

DEAL_II_NAMESPACE_CLOSE

#endif
