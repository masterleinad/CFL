// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 by the deal.II authors
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

#ifndef dealii__mg_transfer_matrix_free_h
#define dealii__mg_transfer_matrix_free_h

#include <deal.II/base/config.h>

#include <deal.II/base/mg_level_object.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <dealii/mg_transfer.h>

#include <deal.II/dofs/dof_handler.h>

DEAL_II_NAMESPACE_OPEN

/*!@addtogroup mg */
/*@{*/

/**
 * Implementation of the MGTransferBase interface for which the transfer
 * operations is implemented in a matrix-free way based on the interpolation
 * matrices of the underlying finite element. This requires considerably less
 * memory than MGTransferPrebuilt and can also be considerably faster than
 * that variant.
 *
 * This class currently only works for tensor-product finite elements based on
 * FE_Q and FE_DGQ elements, including systems involving multiple components
 * of one of these elements. Systems with different elements or other elements
 * are currently not implemented.
 *
 * @author Martin Kronbichler
 * @date 2016
 */
template <int dim, typename Number>
class MGTransferBlockMatrixFree
  : public MGLevelGlobalTransfer<LinearAlgebra::distributed::BlockVector<Number>>
{
public:
  /**
   * Constructor without constraint matrices. Use this constructor only with
   * discontinuous finite elements or with no local refinement.
   */
  MGTransferBlockMatrixFree();

  /**
   * Constructor with constraints. Equivalent to the default constructor
   * followed by initialize_constraints().
   */
  MGTransferBlockMatrixFree(const std::vector<MGConstrainedDoFs>& mg_constrained_dofs);

  /**
   * Destructor.
   */
  virtual ~MGTransferBlockMatrixFree();

  /**
   * Initialize the constraints to be used in build().
   */
  void initialize_constraints(const std::vector<MGConstrainedDoFs>& mg_constrained_dofs);

  /**
   * Reset the object to the state it had right after the default constructor.
   */
  void clear();

  /**
   * Actually build the information for the prolongation for each level.
   */
  void build(const std::vector<DoFHandler<dim, dim>*>& mg_dof);

  /**
   * Prolongate a vector from level <tt>to_level-1</tt> to level
   * <tt>to_level</tt> using the embedding matrices of the underlying finite
   * element. The previous content of <tt>dst</tt> is overwritten.
   *
   * @param to_level The index of the level to prolongate to, which is the
   * level of @p dst.
   *
   * @param src is a vector with as many elements as there are degrees of
   * freedom on the coarser level involved.
   *
   * @param dst has as many elements as there are degrees of freedom on the
   * finer level.
   */
  virtual void prolongate(const unsigned int to_level,
                          LinearAlgebra::distributed::BlockVector<Number>& dst,
                          const LinearAlgebra::distributed::BlockVector<Number>& src) const;

  /**
   * Restrict a vector from level <tt>from_level</tt> to level
   * <tt>from_level-1</tt> using the transpose operation of the prolongate()
   * method. If the region covered by cells on level <tt>from_level</tt> is
   * smaller than that of level <tt>from_level-1</tt> (local refinement), then
   * some degrees of freedom in <tt>dst</tt> are active and will not be
   * altered. For the other degrees of freedom, the result of the restriction
   * is added.
   *
   * @param from_level The index of the level to restrict from, which is the
   * level of @p src.
   *
   * @param src is a vector with as many elements as there are degrees of
   * freedom on the finer level involved.
   *
   * @param dst has as many elements as there are degrees of freedom on the
   * coarser level.
   */
  virtual void restrict_and_add(const unsigned int from_level,
                                LinearAlgebra::distributed::BlockVector<Number>& dst,
                                const LinearAlgebra::distributed::BlockVector<Number>& src) const;

  /**
   * Finite element does not provide prolongation matrices.
   */
  DeclException0(ExcNoProlongation);

  /**
   * Memory used by this object.
   */
  std::size_t memory_consumption() const;

private:
  /**
   * Stores the degree of the finite element contained in the DoFHandler
   * passed to build(). The selection of the computational kernel is based on
   * this number.
   */
  unsigned int fe_degree;

  /**
   * Stores whether the element is continuous and there is a joint degree of
   * freedom in the center of the 1D line.
   */
  bool element_is_continuous;

  /**
   * Stores the number of components in the finite element contained in the
   * DoFHandler passed to build().
   */
  unsigned int n_components;

  /**
   * Stores the number of degrees of freedom on all child cells. It is
   * <tt>2<sup>dim</sup>*fe.dofs_per_cell</tt> for DG elements and somewhat
   * less for continuous elements.
   */
  unsigned int n_child_cell_dofs;

  /**
   * Holds the indices for cells on a given level, extracted from DoFHandler
   * for fast access. All DoF indices on a given level are stored as a plain
   * array (since this class assumes constant DoFs per cell). To index into
   * this array, use the cell number times dofs_per_cell.
   *
   * This array first is arranged such that all locally owned level cells come
   * first (found in the variable n_owned_level_cells) and then other cells
   * necessary for the transfer to the next level.
   */
  std::vector<std::vector<std::vector<unsigned int>>> level_dof_indices;

  /**
   * Stores the connectivity from parent to child cell numbers for each level.
   */
  std::vector<std::vector<std::pair<unsigned int, unsigned int>>> parent_child_connect;

  /**
   * Stores the number of cells owned on a given process (sets the bounds for
   * the worker loops) for each level.
   */
  std::vector<unsigned int> n_owned_level_cells;

  /**
   * Holds the one-dimensional embedding (prolongation) matrix from mother
   * element to all the children.
   */
  AlignedVector<AlignedVector<VectorizedArray<Number>>> prolongation_matrix_1d;

  /**
   * Holds the temporary values for the tensor evaluation
   */
  mutable AlignedVector<VectorizedArray<Number>> evaluation_data;

  /**
   * For continuous elements, restriction is not additive and we need to
   * weight the result at the end of prolongation (and at the start of
   * restriction) by the valence of the degrees of freedom, i.e., on how many
   * elements they appear. We store the data in vectorized form to allow for
   * cheap access. Moreover, we utilize the fact that we only need to store
   * <tt>3<sup>dim</sup></tt> indices.
   *
   * Data is organized in terms of each level (outer vector) and the cells on
   * each level (inner vector).
   */
  std::vector<AlignedVector<VectorizedArray<Number>>> weights_on_refined;

  /**
   * Stores the local indices of Dirichlet boundary conditions on cells for
   * all levels (outer index), the cells within the levels (second index), and
   * the indices on the cell (inner index).
   */
  std::vector<std::vector<std::vector<unsigned short>>> dirichlet_indices;

  /**
   * Performs templated prolongation operation
   */
  template <int degree>
  void do_prolongate_add(const unsigned int to_level,
                         LinearAlgebra::distributed::BlockVector<Number>& dst,
                         const LinearAlgebra::distributed::BlockVector<Number>& src) const;

  /**
   * Performs templated restriction operation
   */
  template <int degree>
  void do_restrict_add(const unsigned int from_level,
                       LinearAlgebra::distributed::BlockVector<Number>& dst,
                       const LinearAlgebra::distributed::BlockVector<Number>& src) const;
};

/*@}*/

DEAL_II_NAMESPACE_CLOSE

#endif
// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 by the deal.II authors
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

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_internal.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <algorithm>

DEAL_II_NAMESPACE_OPEN

template <int dim, typename Number>
MGTransferBlockMatrixFree<dim, Number>::MGTransferBlockMatrixFree()
  : fe_degree(0)
  , element_is_continuous(false)
  , n_components(0)
  , n_child_cell_dofs(0)
{
}

template <int dim, typename Number>
MGTransferBlockMatrixFree<dim, Number>::MGTransferBlockMatrixFree(
  const std::vector<MGConstrainedDoFs>& mg_c)
  : fe_degree(0)
  , element_is_continuous(false)
  , n_components(0)
  , n_child_cell_dofs(0)
{
  this->mg_constrained_dofs = &mg_c;
}

template <int dim, typename Number>
MGTransferBlockMatrixFree<dim, Number>::~MGTransferBlockMatrixFree()
{
}

template <int dim, typename Number>
void
MGTransferBlockMatrixFree<dim, Number>::initialize_constraints(
  const std::vector<MGConstrainedDoFs>& mg_c)
{
  this->mg_constrained_dofs = &mg_c;
}

template <int dim, typename Number>
void
MGTransferBlockMatrixFree<dim, Number>::clear()
{
  this->MGLevelGlobalTransfer<LinearAlgebra::distributed::BlockVector<Number>>::clear();
  fe_degree = 0;
  element_is_continuous = false;
  n_components = 0;
  n_child_cell_dofs = 0;
  level_dof_indices.clear();
  parent_child_connect.clear();
  n_owned_level_cells.clear();
  prolongation_matrix_1d.clear();
  evaluation_data.clear();
  weights_on_refined.clear();
}

template <int dim, typename Number>
void
MGTransferBlockMatrixFree<dim, Number>::build(const std::vector<DoFHandler<dim, dim>*>& mg_dof)
{
  this->fill_and_communicate_copy_indices(mg_dof);

  for (unsigned int k = 0; k < mg_dof.size(); ++k)
  {

    std::vector<std::vector<Number>> weights_unvectorized;

    internal::MGTransfer::ElementInfo<Number> elem_info;

    internal::MGTransfer::setup_transfer<dim, Number>(
      *(mg_dof[k]),
      (const dealii::MGConstrainedDoFs*)(&((*(this->mg_constrained_dofs))[k])),
      elem_info,
      level_dof_indices[k],
      parent_child_connect,
      n_owned_level_cells,
      dirichlet_indices,
      weights_unvectorized,
      this->copy_indices_global_mine[k],
      this->ghosted_level_vector[k]);
    // unpack element info data
    fe_degree = elem_info.fe_degree;
    element_is_continuous = elem_info.element_is_continuous;
    n_components = elem_info.n_components;
    n_child_cell_dofs = elem_info.n_child_cell_dofs;

    // duplicate and put into vectorized array
    prolongation_matrix_1d.resize(elem_info.prolongation_matrix_1d.size());
    for (unsigned int i = 0; i < elem_info.prolongation_matrix_1d.size(); i++)
      prolongation_matrix_1d[i] = elem_info.prolongation_matrix_1d[i];

    // reshuffle into aligned vector of vectorized arrays
    const unsigned int vec_size = VectorizedArray<Number>::n_array_elements;
    const unsigned int n_levels = mg_dof.get_triangulation().n_global_levels();

    const unsigned int n_weights_per_cell = Utilities::fixed_power<dim>(3);
    weights_on_refined.resize(n_levels - 1);
    for (unsigned int level = 1; level < n_levels; ++level)
    {
      weights_on_refined[level - 1].resize(
        ((n_owned_level_cells[level - 1] + vec_size - 1) / vec_size) * n_weights_per_cell);

      for (unsigned int c = 0; c < n_owned_level_cells[level - 1]; ++c)
      {
        const unsigned int comp = c / vec_size;
        const unsigned int v = c % vec_size;
        for (unsigned int i = 0; i < n_weights_per_cell; ++i)
        {

          weights_on_refined[level - 1][comp * n_weights_per_cell + i][v] =
            weights_unvectorized[level - 1][c * n_weights_per_cell + i];
        }
      }
    }

    evaluation_data.resize(3 * n_child_cell_dofs);
  }
}

template <int dim, typename Number>
void
MGTransferBlockMatrixFree<dim, Number>::prolongate(
  const unsigned int to_level, LinearAlgebra::distributed::BlockVector<Number>& dst,
  const LinearAlgebra::distributed::BlockVector<Number>& src) const
{
  Assert((to_level >= 1) && (to_level <= level_dof_indices.size()),
         ExcIndexRange(to_level, 1, level_dof_indices.size() + 1));

  AssertDimension(this->ghosted_level_vector[to_level].local_size(), dst.local_size());
  AssertDimension(this->ghosted_level_vector[to_level - 1].local_size(), src.local_size());

  this->ghosted_level_vector[to_level - 1] = src;
  this->ghosted_level_vector[to_level - 1].update_ghost_values();
  this->ghosted_level_vector[to_level] = 0.;

  // the implementation in do_prolongate_add is templated in the degree of the
  // element (for efficiency reasons), so we need to find the appropriate
  // kernel here...
  if (fe_degree == 0)
    do_prolongate_add<0>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 1)
    do_prolongate_add<1>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 2)
    do_prolongate_add<2>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 3)
    do_prolongate_add<3>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 4)
    do_prolongate_add<4>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 5)
    do_prolongate_add<5>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 6)
    do_prolongate_add<6>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 7)
    do_prolongate_add<7>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 8)
    do_prolongate_add<8>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 9)
    do_prolongate_add<9>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else if (fe_degree == 10)
    do_prolongate_add<10>(
      to_level, this->ghosted_level_vector[to_level], this->ghosted_level_vector[to_level - 1]);
  else
    AssertThrow(false, ExcNotImplemented("Only degrees 0 up to 10 implemented."));

  this->ghosted_level_vector[to_level].compress(VectorOperation::add);
  dst = this->ghosted_level_vector[to_level];
}

template <int dim, typename Number>
void
MGTransferBlockMatrixFree<dim, Number>::restrict_and_add(
  const unsigned int from_level, LinearAlgebra::distributed::BlockVector<Number>& dst,
  const LinearAlgebra::distributed::BlockVector<Number>& src) const
{
  Assert((from_level >= 1) && (from_level <= level_dof_indices.size()),
         ExcIndexRange(from_level, 1, level_dof_indices.size() + 1));

  AssertDimension(this->ghosted_level_vector[from_level].local_size(), src.local_size());
  AssertDimension(this->ghosted_level_vector[from_level - 1].local_size(), dst.local_size());

  this->ghosted_level_vector[from_level] = src;
  this->ghosted_level_vector[from_level].update_ghost_values();
  this->ghosted_level_vector[from_level - 1] = 0.;

  if (fe_degree == 0)
    do_restrict_add<0>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 1)
    do_restrict_add<1>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 2)
    do_restrict_add<2>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 3)
    do_restrict_add<3>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 4)
    do_restrict_add<4>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 5)
    do_restrict_add<5>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 6)
    do_restrict_add<6>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 7)
    do_restrict_add<7>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 8)
    do_restrict_add<8>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 9)
    do_restrict_add<9>(from_level,
                       this->ghosted_level_vector[from_level - 1],
                       this->ghosted_level_vector[from_level]);
  else if (fe_degree == 10)
    do_restrict_add<10>(from_level,
                        this->ghosted_level_vector[from_level - 1],
                        this->ghosted_level_vector[from_level]);
  else
    AssertThrow(false, ExcNotImplemented("Only degrees 0 up to 10 implemented."));

  this->ghosted_level_vector[from_level - 1].compress(VectorOperation::add);
  dst += this->ghosted_level_vector[from_level - 1];
}

namespace
{
template <int dim, typename Eval, typename Number, bool prolongate>
void
perform_tensorized_op(const Eval& evaluator, const unsigned int n_child_cell_dofs,
                      const unsigned int n_components,
                      AlignedVector<VectorizedArray<Number>>& evaluation_data)
{
  AssertDimension(n_components * Eval::n_q_points, n_child_cell_dofs);
  VectorizedArray<Number>* t0 = &evaluation_data[0];
  VectorizedArray<Number>* t1 = &evaluation_data[n_child_cell_dofs];
  VectorizedArray<Number>* t2 = &evaluation_data[2 * n_child_cell_dofs];

  for (unsigned int c = 0; c < n_components; ++c)
  {
    // for the prolongate case, we go from dofs (living on the parent cell) to
    // quads (living on all children) in the FEEvaluation terminology
    if (dim == 1)
      evaluator.template values<0, prolongate, false>(t0, t2);
    else if (dim == 2)
    {
      evaluator.template values<0, prolongate, false>(t0, t1);
      evaluator.template values<1, prolongate, false>(t1, t2);
    }
    else if (dim == 3)
    {
      evaluator.template values<0, prolongate, false>(t0, t2);
      evaluator.template values<1, prolongate, false>(t2, t1);
      evaluator.template values<2, prolongate, false>(t1, t2);
    }
    else
      Assert(false, ExcNotImplemented());
    if (prolongate)
    {
      t0 += Eval::dofs_per_cell;
      t2 += Eval::n_q_points;
    }
    else
    {
      t0 += Eval::n_q_points;
      t2 += Eval::dofs_per_cell;
    }
  }
}

template <int dim, int degree, typename Number>
void
weight_dofs_on_child(const VectorizedArray<Number>* weights, const unsigned int n_components,
                     VectorizedArray<Number>* data)
{
  Assert(degree > 0, ExcNotImplemented());
  const int loop_length = 2 * degree + 1;
  unsigned int degree_to_3[loop_length];
  degree_to_3[0] = 0;
  for (int i = 1; i < loop_length - 1; ++i)
    degree_to_3[i] = 1;
  degree_to_3[loop_length - 1] = 2;
  for (unsigned int c = 0; c < n_components; ++c)
    for (int k = 0; k < (dim > 2 ? loop_length : 1); ++k)
      for (int j = 0; j < (dim > 1 ? loop_length : 1); ++j)
      {
        const unsigned int shift = 9 * degree_to_3[k] + 3 * degree_to_3[j];
        data[0] *= weights[shift];
        // loop bound as int avoids compiler warnings in case loop_length
        // == 1 (polynomial degree 0)
        for (int i = 1; i < loop_length - 1; ++i)
          data[i] *= weights[shift + 1];
        data[loop_length - 1] *= weights[shift + 2];
        data += loop_length;
      }
}
}

template <int dim, typename Number>
template <int degree>
void
MGTransferBlockMatrixFree<dim, Number>::do_prolongate_add(
  const unsigned int to_level, LinearAlgebra::distributed::BlockVector<Number>& dst,
  const LinearAlgebra::distributed::BlockVector<Number>& src) const
{
  const unsigned int vec_size = VectorizedArray<Number>::n_array_elements;
  const unsigned int n_child_dofs_1d = 2 * (fe_degree + 1) - element_is_continuous;
  const unsigned int n_scalar_cell_dofs = Utilities::fixed_power<dim>(n_child_dofs_1d);
  const unsigned int three_to_dim = Utilities::fixed_int_power<3, dim>::value;

  for (unsigned int cell = 0; cell < n_owned_level_cells[to_level - 1]; cell += vec_size)
  {
    const unsigned int n_chunks = cell + vec_size > n_owned_level_cells[to_level - 1]
                                    ? n_owned_level_cells[to_level - 1] - cell
                                    : vec_size;

    // read from source vector
    for (unsigned int v = 0; v < n_chunks; ++v)
    {
      const unsigned int shift = internal::MGTransfer::compute_shift_within_children<dim>(
        parent_child_connect[to_level - 1][cell + v].second,
        degree + 1 - element_is_continuous,
        degree);
      const unsigned int* indices =
        &level_dof_indices[to_level - 1]
                          [parent_child_connect[to_level - 1][cell + v].first * n_child_cell_dofs +
                           shift];
      for (unsigned int c = 0, m = 0; c < n_components; ++c)
      {
        for (unsigned int k = 0; k < (dim > 2 ? (degree + 1) : 1); ++k)
          for (unsigned int j = 0; j < (dim > 1 ? (degree + 1) : 1); ++j)
            for (unsigned int i = 0; i < (degree + 1); ++i, ++m)
              evaluation_data[m][v] = src.local_element(
                indices[c * n_scalar_cell_dofs + k * n_child_dofs_1d * n_child_dofs_1d +
                        j * n_child_dofs_1d + i]);

        // apply Dirichlet boundary conditions on parent cell
        for (std::vector<unsigned short>::const_iterator i =
               dirichlet_indices[to_level - 1][cell + v].begin();
             i != dirichlet_indices[to_level - 1][cell + v].end();
             ++i)
          evaluation_data[*i][v] = 0.;
      }
    }

    // perform tensorized operation
    if (element_is_continuous)
    {
      AssertDimension(prolongation_matrix_1d.size(), (2 * degree + 1) * (degree + 1));
      typedef internal::EvaluatorTensorProduct<internal::evaluate_general,
                                               dim,
                                               degree,
                                               2 * degree + 1,
                                               VectorizedArray<Number>>
        Evaluator;
      Evaluator evaluator(prolongation_matrix_1d, prolongation_matrix_1d, prolongation_matrix_1d);
      perform_tensorized_op<dim, Evaluator, Number, true>(
        evaluator, n_child_cell_dofs, n_components, evaluation_data);
      weight_dofs_on_child<dim, degree, Number>(
        &weights_on_refined[to_level - 1][(cell / vec_size) * three_to_dim],
        n_components,
        &evaluation_data[2 * n_child_cell_dofs]);
    }
    else
    {
      AssertDimension(prolongation_matrix_1d.size(), 2 * (degree + 1) * (degree + 1));
      typedef internal::EvaluatorTensorProduct<internal::evaluate_general,
                                               dim,
                                               degree,
                                               2 * (degree + 1),
                                               VectorizedArray<Number>>
        Evaluator;
      Evaluator evaluator(prolongation_matrix_1d, prolongation_matrix_1d, prolongation_matrix_1d);
      perform_tensorized_op<dim, Evaluator, Number, true>(
        evaluator, n_child_cell_dofs, n_components, evaluation_data);
    }

    // write into dst vector
    const unsigned int* indices = &level_dof_indices[to_level][cell * n_child_cell_dofs];
    for (unsigned int v = 0; v < n_chunks; ++v)
    {
      for (unsigned int i = 0; i < n_child_cell_dofs; ++i)
        dst.local_element(indices[i]) += evaluation_data[2 * n_child_cell_dofs + i][v];
      indices += n_child_cell_dofs;
    }
  }
}

template <int dim, typename Number>
template <int degree>
void
MGTransferBlockMatrixFree<dim, Number>::do_restrict_add(
  const unsigned int from_level, LinearAlgebra::distributed::BlockVector<Number>& dst,
  const LinearAlgebra::distributed::BlockVector<Number>& src) const
{
  const unsigned int vec_size = VectorizedArray<Number>::n_array_elements;
  const unsigned int n_child_dofs_1d = 2 * (fe_degree + 1) - element_is_continuous;
  const unsigned int n_scalar_cell_dofs = Utilities::fixed_power<dim>(n_child_dofs_1d);
  const unsigned int three_to_dim = Utilities::fixed_int_power<3, dim>::value;

  for (unsigned int cell = 0; cell < n_owned_level_cells[from_level - 1]; cell += vec_size)
  {
    const unsigned int n_chunks = cell + vec_size > n_owned_level_cells[from_level - 1]
                                    ? n_owned_level_cells[from_level - 1] - cell
                                    : vec_size;

    // read from source vector
    {
      const unsigned int* indices = &level_dof_indices[from_level][cell * n_child_cell_dofs];
      for (unsigned int v = 0; v < n_chunks; ++v)
      {
        for (unsigned int i = 0; i < n_child_cell_dofs; ++i)
          evaluation_data[i][v] = src.local_element(indices[i]);
        indices += n_child_cell_dofs;
      }
    }

    // perform tensorized operation
    if (element_is_continuous)
    {
      AssertDimension(prolongation_matrix_1d.size(), (2 * degree + 1) * (degree + 1));
      typedef internal::EvaluatorTensorProduct<internal::evaluate_general,
                                               dim,
                                               degree,
                                               2 * degree + 1,
                                               VectorizedArray<Number>>
        Evaluator;
      Evaluator evaluator(prolongation_matrix_1d, prolongation_matrix_1d, prolongation_matrix_1d);
      weight_dofs_on_child<dim, degree, Number>(
        &weights_on_refined[from_level - 1][(cell / vec_size) * three_to_dim],
        n_components,
        &evaluation_data[0]);
      perform_tensorized_op<dim, Evaluator, Number, false>(
        evaluator, n_child_cell_dofs, n_components, evaluation_data);
    }
    else
    {
      AssertDimension(prolongation_matrix_1d.size(), 2 * (degree + 1) * (degree + 1));
      typedef internal::EvaluatorTensorProduct<internal::evaluate_general,
                                               dim,
                                               degree,
                                               2 * (degree + 1),
                                               VectorizedArray<Number>>
        Evaluator;
      Evaluator evaluator(prolongation_matrix_1d, prolongation_matrix_1d, prolongation_matrix_1d);
      perform_tensorized_op<dim, Evaluator, Number, false>(
        evaluator, n_child_cell_dofs, n_components, evaluation_data);
    }

    // write into dst vector
    for (unsigned int v = 0; v < n_chunks; ++v)
    {
      const unsigned int shift = internal::MGTransfer::compute_shift_within_children<dim>(
        parent_child_connect[from_level - 1][cell + v].second,
        degree + 1 - element_is_continuous,
        degree);
      AssertIndexRange(parent_child_connect[from_level - 1][cell + v].first * n_child_cell_dofs +
                         n_child_cell_dofs - 1,
                       level_dof_indices[from_level - 1].size());
      const unsigned int* indices =
        &level_dof_indices[from_level - 1][parent_child_connect[from_level - 1][cell + v].first *
                                             n_child_cell_dofs +
                                           shift];
      for (unsigned int c = 0, m = 0; c < n_components; ++c)
      {
        // apply Dirichlet boundary conditions on parent cell
        for (std::vector<unsigned short>::const_iterator i =
               dirichlet_indices[from_level - 1][cell + v].begin();
             i != dirichlet_indices[from_level - 1][cell + v].end();
             ++i)
          evaluation_data[2 * n_child_cell_dofs + (*i)][v] = 0.;

        for (unsigned int k = 0; k < (dim > 2 ? (degree + 1) : 1); ++k)
          for (unsigned int j = 0; j < (dim > 1 ? (degree + 1) : 1); ++j)
            for (unsigned int i = 0; i < (degree + 1); ++i, ++m)
              dst.local_element(
                indices[c * n_scalar_cell_dofs + k * n_child_dofs_1d * n_child_dofs_1d +
                        j * n_child_dofs_1d + i]) += evaluation_data[2 * n_child_cell_dofs + m][v];
      }
    }
  }
}

template <int dim, typename Number>
std::size_t
MGTransferBlockMatrixFree<dim, Number>::memory_consumption() const
{
  std::size_t memory =
    MGLevelGlobalTransfer<LinearAlgebra::distributed::Vector<Number>>::memory_consumption();
  memory += MemoryConsumption::memory_consumption(level_dof_indices);
  memory += MemoryConsumption::memory_consumption(parent_child_connect);
  memory += MemoryConsumption::memory_consumption(n_owned_level_cells);
  memory += MemoryConsumption::memory_consumption(prolongation_matrix_1d);
  memory += MemoryConsumption::memory_consumption(evaluation_data);
  memory += MemoryConsumption::memory_consumption(weights_on_refined);
  memory += MemoryConsumption::memory_consumption(dirichlet_indices);
  return memory;
}

DEAL_II_NAMESPACE_CLOSE
