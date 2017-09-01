//---------------------------------------------------------------------------
//    $Id: solution.h 67 2015-03-03 11:34:17Z kronbichler $
//    Version: $Name$
//
//    Copyright (C) 2013 - 2014 by Katharina Kormann and Martin Kronbichler
//
//---------------------------------------------------------------------------

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;

template <int dim>
class SolutionBase
{
protected:
  static const unsigned int n_source_centers = 3;
  static const Point<dim> source_centers[n_source_centers];
  static const double width;
};

template <>
const Point<1> SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] =
  { Point<1>(-1.0 / 3.0), Point<1>(0.0), Point<1>(+1.0 / 3.0) };

template <>
const Point<2> SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] =
  { Point<2>(-0.5, +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.5, -0.5) };

template <>
const Point<3> SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] =
  { Point<3>(-0.5, +0.5, 0.25), Point<3>(-0.6, -0.5, -0.125), Point<3>(+0.5, -0.5, 0.5) };

template <int dim>
const double SolutionBase<dim>::width = 1. / 3.;

template <int dim>
class Solution : public Function<dim>, protected SolutionBase<dim>
{
public:
  Solution()
    : Function<dim>()
  {
  }

  virtual double value(const Point<dim>& p, const unsigned int component = 0) const;

  virtual Tensor<1, dim> gradient(const Point<dim>& p, const unsigned int component = 0) const;
};

template <int dim>
double
Solution<dim>::value(const Point<dim>& p, const unsigned int) const
{
  const double pi = numbers::PI;
  double return_value = 0;
  for (unsigned int i = 0; i < this->n_source_centers; ++i)
  {
    const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
    return_value += std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
  }

  return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * pi) * this->width);
}

template <int dim>
Tensor<1, dim>
Solution<dim>::gradient(const Point<dim>& p, const unsigned int) const
{
  const double pi = numbers::PI;
  Tensor<1, dim> return_value;

  for (unsigned int i = 0; i < this->n_source_centers; ++i)
  {
    const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

    return_value +=
      (-2 / (this->width * this->width) *
       std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) * x_minus_xi);
  }

  return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * pi) * this->width);
}

template <int dim>
class RightHandSide : public Function<dim>, protected SolutionBase<dim>
{
public:
  RightHandSide()
    : Function<dim>()
  {
  }

  virtual double value(const Point<dim>& p, const unsigned int component = 0) const;
};

template <int dim>
double
RightHandSide<dim>::value(const Point<dim>& p, const unsigned int) const
{
  const double pi = numbers::PI;
  double return_value = 0;
  for (unsigned int i = 0; i < this->n_source_centers; ++i)
  {
    const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

    // The first contribution is the
    // Laplacian:
    return_value += ((2 * dim - 4 * x_minus_xi.norm_square() / (this->width * this->width)) /
                     (this->width * this->width) *
                     std::exp(-x_minus_xi.norm_square() / (this->width * this->width)));
  }

  return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * pi) * this->width);
}

template <int dim>
class SolutionNegGrad : public Function<dim>, protected SolutionBase<dim>
{
public:
  SolutionNegGrad()
    : Function<dim>(dim)
  {
  }

  virtual void
  vector_value(const Point<dim>& p, Vector<double>& v) const
  {
    AssertDimension(v.size(), dim);
    Tensor<1, dim> grad = solution.gradient(p);
    for (unsigned int d = 0; d < dim; ++d)
      v[d] = -grad[d];
  }

private:
  Solution<dim> solution;
};
