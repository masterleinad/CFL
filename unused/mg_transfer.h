// ---------------------------------------------------------------------
//
// Copyright (C) 2001 - 2017 by the deal.II authors
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

#ifndef dealii__mg_transfer_h
#define dealii__mg_transfer_h

#include <deal.II/base/config.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/lac/vector_memory.h>

#include <deal.II/base/mg_level_object.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/base/std_cxx11/shared_ptr.h>

DEAL_II_NAMESPACE_OPEN

template <typename VectorType>
class MGLevelGlobalTransfer : public MGTransferBase<VectorType>
{
};

/**
 * Implementation of transfer between the global vectors and the multigrid
 * levels for use in the derived class MGTransferPrebuilt and other classes.
 * This class is a specialization for the case of
 * LinearAlgebra::distributed::Vector that requires a few different calling
 * routines as compared to the %parallel vectors in the PETScWrappers and
 * TrilinosWrappers namespaces.
 *
 * @author Martin Kronbichler
 * @date 2016
 */
template <typename Number>
class MGLevelGlobalTransfer<LinearAlgebra::distributed::BlockVector<Number>>
  : public MGTransferBase<LinearAlgebra::distributed::BlockVector<Number>>
{
public:
  template <int dim, int spacedim>
  void build_matrices(const std::vector<DoFHandler<dim, spacedim>>& mg_dof);

  /**
   * Reset the object to the state it had right after the default constructor.
   */
  void clear();

  /**
   * Transfer from a vector on the global grid to vectors defined on each of
   * the levels separately, i.a. an @p MGVector.
   */
  template <int dim, typename Number2, int spacedim>
  void copy_to_mg(const std::vector<DoFHandler<dim, spacedim>*>& mg_dof,
                  MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>>& dst,
                  const LinearAlgebra::distributed::BlockVector<Number2>& src) const;

  /**
   * Transfer from multi-level vector to normal vector.
   *
   * Copies data from active portions of an MGVector into the respective
   * positions of a <tt>Vector<number></tt>. In order to keep the result
   * consistent, constrained degrees of freedom are set to zero.
   */
  template <int dim, typename Number2, int spacedim>
  void copy_from_mg(
    const std::vector<DoFHandler<dim, spacedim>*>& mg_dof,
    LinearAlgebra::distributed::BlockVector<Number2>& dst,
    const MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>>& src) const;

  /**
   * Add a multi-level vector to a normal vector.
   *
   * Works as the previous function, but probably not for continuous elements.
   */
  template <int dim, typename Number2, int spacedim>
  void copy_from_mg_add(
    const std::vector<DoFHandler<dim, spacedim>*>& mg_dof,
    LinearAlgebra::distributed::BlockVector<Number2>& dst,
    const MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>>& src) const;

  /**
   * If this object operates on BlockVector objects, we need to describe how
   * the individual vector components are mapped to the blocks of a vector.
   * For example, for a Stokes system, we have dim+1 vector components for
   * velocity and pressure, but we may want to use block vectors with only two
   * blocks for all velocities in one block, and the pressure variables in the
   * other.
   *
   * By default, if this function is not called, block vectors have as many
   * blocks as the finite element has vector components. However, this can be
   * changed by calling this function with an array that describes how vector
   * components are to be grouped into blocks. The meaning of the argument is
   * the same as the one given to the DoFTools::count_dofs_per_component
   * function.
   */
  void set_component_to_block_map(const std::vector<unsigned int>& map);

  /**
   * Memory used by this object.
   */
  std::size_t memory_consumption() const;

  /**
   * Print the copy index fields for debugging purposes.
   */
  void print_indices(std::ostream& os) const;

protected:
  /**
   * Internal function to @p fill copy_indices*. Called by derived classes.
   */
  template <int dim, int spacedim>
  void fill_and_communicate_copy_indices(const std::vector<DoFHandler<dim, spacedim>*>& mg_dof);

  /**
   * Sizes of the multi-level vectors.
   */
  std::vector<std::vector<types::global_dof_index>*> sizes;

  /**
   * Mapping for the copy_to_mg() and copy_from_mg() functions. Here only
   * index pairs locally owned is stored.
   *
   * The data is organized as follows: one vector per level. Each element of
   * these vectors contains first the global index, then the level index.
   */
  std::vector<std::vector<std::vector<std::pair<unsigned int, unsigned int>>>> copy_indices;

  /**
   * Additional degrees of freedom for the copy_to_mg() function. These are
   * the ones where the global degree of freedom is locally owned and the
   * level degree of freedom is not.
   *
   * Organization of the data is like for @p copy_indices_mine.
   */
  std::vector<std::vector<std::vector<std::pair<unsigned int, unsigned int>>>>
    copy_indices_global_mine;

  /**
   * Additional degrees of freedom for the copy_from_mg() function. These are
   * the ones where the level degree of freedom is locally owned and the
   * global degree of freedom is not.
   *
   * Organization of the data is like for @p copy_indices_mine.
   */
  std::vector<std::vector<std::vector<std::pair<unsigned int, unsigned int>>>>
    copy_indices_level_mine;

  /**
   * Stores whether the copy operation from the global to the level vector is
   * actually a plain copy to the finest level. This means that the grid has
   * no adaptive refinement and the numbering on the finest multigrid level is
   * the same as in the global case.
   */
  bool perform_plain_copy;

  /**
   * The vector that stores what has been given to the
   * set_component_to_block_map() function.
   */
  std::vector<unsigned int> component_to_block_map;

  /**
   * The mg_constrained_dofs of the level systems.
   */
  const std::vector<MGConstrainedDoFs>* mg_constrained_dofs;

  /**
   * In the function copy_to_mg, we need to access ghosted entries of the
   * global vector for inserting into the level vectors. This vector is
   * populated with those entries.
   */
  mutable LinearAlgebra::distributed::BlockVector<Number> ghosted_global_vector;

  /**
   * In the function copy_from_mg, we access all level vectors with certain
   * ghost entries for inserting the result into a global vector.
   */
  mutable MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>> ghosted_level_vector;
};

DEAL_II_NAMESPACE_CLOSE

#endif
// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2016 by the deal.II authors
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
#include <deal.II/fe/fe.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
//#include <deal.II/multigrid/mg_transfer.templates.h>
#include <deal.II/multigrid/mg_transfer_internal.h>

#include <algorithm>

DEAL_II_NAMESPACE_OPEN

/* - MGLevelGlobalTransfer<LinearAlgebra::distributed::BlockVector<Number> > - */

template <typename Number>
template <int dim, int spacedim>
void
MGLevelGlobalTransfer<LinearAlgebra::distributed::BlockVector<Number>>::
  fill_and_communicate_copy_indices(const std::vector<DoFHandler<dim, spacedim>*>& mg_dof)
{
  const unsigned int n_blocks = mg_dof.size();
  // first go to the usual routine...
  std::vector<std::vector<std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>>
    my_copy_indices(n_blocks);
  std::vector<std::vector<std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>>
    my_copy_indices_global_mine(n_blocks);
  std::vector<std::vector<std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>>
    my_copy_indices_level_mine(n_blocks);

  copy_indices.resize(n_blocks);
  copy_indices_global_mine.resize(n_blocks);
  copy_indices_level_mine.resize(n_blocks);

  for (unsigned int k = 0; k < mg_dof.size(); ++k)
  {
    internal::MGTransfer::fill_copy_indices(*(mg_dof[k]),
                                            mg_constrained_dofs[k],
                                            my_copy_indices[k],
                                            my_copy_indices_global_mine[k],
                                            my_copy_indices_level_mine[k]);

    // get all degrees of freedom that we need read access to in copy_to_mg
    // and copy_from_mg, respectively. We fill an IndexSet once on each level
    // (for the global_mine indices accessing remote level indices) and once
    // globally (for the level_mine indices accessing remote global indices).

    // the variables index_set and level_index_set are going to define the
    // ghost indices of the respective vectors (due to construction, these are
    // precisely the indices that we need)
    const parallel::Triangulation<dim, spacedim>* ptria =
      dynamic_cast<const parallel::Triangulation<dim, spacedim>*>(&mg_dof[k].get_tria());
    const MPI_Comm mpi_communicator = ptria != 0 ? ptria->get_communicator() : MPI_COMM_SELF;

    IndexSet index_set(mg_dof[k].locally_owned_dofs().size());
    std::vector<types::global_dof_index> accessed_indices;
    ghosted_level_vector.resize(0, mg_dof[k].get_triangulation().n_global_levels() - 1);
    std::vector<IndexSet> level_index_set(mg_dof[k].get_triangulation().n_global_levels());

    for (unsigned int l = 0; l < mg_dof.get_triangulation().n_global_levels(); ++l)
    {
      for (unsigned int i = 0; i < my_copy_indices_level_mine[k][l].size(); ++i)
        accessed_indices.push_back(my_copy_indices_level_mine[k][l][i].first);
      std::vector<types::global_dof_index> accessed_level_indices;
      for (unsigned int i = 0; i < my_copy_indices_global_mine[k][l].size(); ++i)
        accessed_level_indices.push_back(my_copy_indices_global_mine[k][l][i].second);
      std::sort(accessed_level_indices.begin(), accessed_level_indices.end());
      level_index_set[l].set_size(mg_dof.locally_owned_mg_dofs(l).size());
      level_index_set[l].add_indices(accessed_level_indices.begin(), accessed_level_indices.end());
      level_index_set[l].compress();
      ghosted_level_vector[l].reinit(
        mg_dof.locally_owned_mg_dofs(l), level_index_set[l], mpi_communicator);
    }
    std::sort(accessed_indices.begin(), accessed_indices.end());
    index_set.add_indices(accessed_indices.begin(), accessed_indices.end());
    index_set.compress();
    ghosted_global_vector.reinit(mg_dof.locally_owned_dofs(), index_set, mpi_communicator);

    // localize the copy indices for faster access. Since all access will be
    // through the ghosted vector in 'data', we can use this (much faster)
    // option
    this->copy_indices.resize(mg_dof.get_triangulation().n_global_levels());
    this->copy_indices_level_mine.resize(mg_dof.get_triangulation().n_global_levels());
    this->copy_indices_global_mine.resize(mg_dof.get_triangulation().n_global_levels());
    for (unsigned int level = 0; level < mg_dof.get_triangulation().n_global_levels(); ++level)
    {
      const Utilities::MPI::Partitioner& global_partitioner =
        *ghosted_global_vector.get_partitioner();
      const Utilities::MPI::Partitioner& level_partitioner =
        *ghosted_level_vector[level].get_partitioner();
      // owned-owned case: the locally owned indices are going to control
      // the local index
      this->copy_indices[level].resize(my_copy_indices[k][level].size());
      for (unsigned int i = 0; i < my_copy_indices[k][level].size(); ++i)
        this->copy_indices[level][i] = std::pair<unsigned int, unsigned int>(
          global_partitioner.global_to_local(my_copy_indices[k][level][i].first),
          level_partitioner.global_to_local(my_copy_indices[k][level][i].second));

      // remote-owned case: the locally owned indices for the level and the
      // ghost dofs for the global indices set the local index
      this->copy_indices_level_mine[level].resize(my_copy_indices_level_mine[k][level].size());
      for (unsigned int i = 0; i < my_copy_indices_level_mine[k][level].size(); ++i)
        this->copy_indices_level_mine[level][i] = std::pair<unsigned int, unsigned int>(
          global_partitioner.global_to_local(my_copy_indices_level_mine[k][level][i].first),
          level_partitioner.global_to_local(my_copy_indices_level_mine[k][level][i].second));

      // owned-remote case: the locally owned indices for the global dofs
      // and the ghost dofs for the level indices set the local index
      this->copy_indices_global_mine[level].resize(my_copy_indices_global_mine[k][level].size());
      for (unsigned int i = 0; i < my_copy_indices_global_mine[k][level].size(); ++i)
        this->copy_indices_global_mine[level][i] = std::pair<unsigned int, unsigned int>(
          global_partitioner.global_to_local(my_copy_indices_global_mine[k][level][i].first),
          level_partitioner.global_to_local(my_copy_indices_global_mine[k][level][i].second));
    }

    perform_plain_copy =
      this->copy_indices.back().size() == mg_dof.locally_owned_dofs().n_elements();
    if (perform_plain_copy)
    {
      AssertDimension(this->copy_indices_global_mine[k].back().size(), 0);
      AssertDimension(this->copy_indices_level_mine[k].back().size(), 0);

      // check whether there is a renumbering of degrees of freedom on
      // either the finest level or the global dofs, which means that we
      // cannot apply a plain copy
      for (unsigned int i = 0; i < this->copy_indices[k].back().size(); ++i)
        if (this->copy_indices[k].back()[i].first != this->copy_indices[k].back()[i].second)
        {
          perform_plain_copy = false;
          break;
        }
    }
    perform_plain_copy =
      Utilities::MPI::min(static_cast<int>(perform_plain_copy), mpi_communicator);

    // if we do a plain copy, no need to hold additional ghosted vectors
    if (perform_plain_copy)
    {
      ghosted_global_vector.reinit(0);
      ghosted_level_vector.resize(0, 0);
    }
  }
}

template <typename Number>
void
MGLevelGlobalTransfer<LinearAlgebra::distributed::BlockVector<Number>>::clear()
{
  sizes.resize(0);
  copy_indices.clear();
  copy_indices_global_mine.clear();
  copy_indices_level_mine.clear();
  component_to_block_map.resize(0);
  mg_constrained_dofs = 0;
  ghosted_global_vector.reinit(0);
  ghosted_level_vector.resize(0, 0);
}

template <typename Number>
void
MGLevelGlobalTransfer<LinearAlgebra::distributed::BlockVector<Number>>::print_indices(
  std::ostream& os) const
{
  Assert(copy_indices.size() == copy_indices_level_mine.size(), ExcInternalError());
  Assert(copy_indices.size() == copy_indices_global_mine.size(), ExcInternalError());
  for (unsigned int k = 0; k < copy_indices[k].size(); ++k)
  {
    for (unsigned int level = 0; level < copy_indices[k].size(); ++level)
    {
      for (unsigned int i = 0; i < copy_indices[k][level].size(); ++i)
        os << "copy_indices[" << level << "]\t" << copy_indices[k][level][i].first << '\t'
           << copy_indices[k][level][i].second << std::endl;
    }

    for (unsigned int level = 0; level < copy_indices_level_mine[k].size(); ++level)
    {
      for (unsigned int i = 0; i < copy_indices_level_mine[k][level].size(); ++i)
        os << "copy_ifrom  [" << level << "]\t" << copy_indices_level_mine[k][level][i].first
           << '\t' << copy_indices_level_mine[k][level][i].second << std::endl;
    }
    for (unsigned int level = 0; level < copy_indices_global_mine[k].size(); ++level)
    {
      for (unsigned int i = 0; i < copy_indices_global_mine[k][level].size(); ++i)
        os << "copy_ito    [" << level << "]\t" << copy_indices_global_mine[k][level][i].first
           << '\t' << copy_indices_global_mine[k][level][i].second << std::endl;
    }
  }
}

template <typename Number>
std::size_t
MGLevelGlobalTransfer<LinearAlgebra::distributed::BlockVector<Number>>::memory_consumption() const
{
  std::size_t result = sizeof(*this);
  result += MemoryConsumption::memory_consumption(sizes);
  result += MemoryConsumption::memory_consumption(copy_indices);
  result += MemoryConsumption::memory_consumption(copy_indices_global_mine);
  result += MemoryConsumption::memory_consumption(copy_indices_level_mine);
  result += ghosted_global_vector.memory_consumption();
  for (unsigned int i = ghosted_level_vector.min_level(); i <= ghosted_level_vector.max_level();
       ++i)
    result += ghosted_level_vector[i].memory_consumption();

  return result;
}

DEAL_II_NAMESPACE_CLOSE
