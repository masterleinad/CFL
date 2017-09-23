#ifndef MESHWORKER_FORMS_H
#define MESHWORKER_FORMS_H

#include <cfl/base/forms.h>

namespace CFL::dealii::MeshWorker
{
template <class TEST, class EXPR, FormKind kind_of_form>
void
anchor(
  const Base::Form<TEST, EXPR, kind_of_form>& form,
  const ::dealii::MeshWorker::IntegrationInfo<TEST::TensorTraits::dim, TEST::TensorTraits::dim>& ii,
  const ::dealii::MeshWorker::LocalIntegrator<TEST::TensorTraits::dim>& li)
{
  form.expr.anchor(ii, li);
}

template <class TEST, class EXPR, FormKind kind_of_form>
void
reinit(
  const Base::Form<TEST, EXPR, kind_of_form>& form,
  const ::dealii::MeshWorker::IntegrationInfo<TEST::TensorTraits::dim, TEST::TensorTraits::dim>& ii)
{
  form.test.reinit(ii);
}

template <class T, int dim = T::TensorTraits::dim>
std::enable_if<Traits::needs_anchor<T>::type>
anchor(const T& t, const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
       const ::dealii::MeshWorker::LocalIntegrator<dim>& li)
{
  t.anchor(ii, li);
}

template <class T, int dim = T::TensorTraits::dim>
std::enable_if<Traits::is_unary_operator<T>::type>
anchor(const T& t, const ::dealii::MeshWorker::IntegrationInfo<dim, dim>& ii,
       const ::dealii::MeshWorker::LocalIntegrator<dim>& li)
{
  anchor(t.base, ii, li);
}
}

#endif // Base::FormS_H
