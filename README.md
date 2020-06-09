# README

## Requirements on the C++ Form Language

- Human readable
- Allows for the definition of intermediate objects and "functions"
- Evaluated only inside a hardware dependent function
- `std` functions of its parameters
- vector valued elements
- arbitrary number type
- systems of equations with combinations of vector-valued elements

## Data exchange mechanisms

Finite element function objects are described by a local shape
function class and an object obtaining function values from a global
finite element vector through `gather`. The shape function class has
to know how to evaluate linear combinations in a single point. It also
has to be able to obtain these points inside the integration loop.

A test function object must be able to compute inner products of a
precomputed expression with a test function or its derivatives.

At least the test and shape function objects for a single part of the
form rely on the same quadrature. They must use the same quadrature point.
How do we synchronize? The following optimizations should be possible

- function values and test functions (1D) precomputed
- recursive definition of polynomials computed  on the fly
- "spectral" evaluation with quadrature in roots
- different quadrature for different parts of the form?


## Examples for application code

These are examples for possible forms of application codes. We develop
them around the examples of

- incompressible, nonstationary Navier-Stokes, Newton method
- chemical diffusion-reaction systems
- radiative transfer
- magneto-hydrodynamics

All codes are for a function taking the necessary input parameters,
building the form description, and then calling a generic function
`integrate`, which does the actual work.


### Incompressible, nonstationary Navier-Stokes

Solution by implicit Euler scheme and Picard iteration. Code is needed
for the nonlinear residual and matrix-vector multiplication for the linearized problem.

#### Code for Newton residual

Incoming data:

- `un_vector`, `pn_vector`: global finite element vectors, velocity and pressure at the current Newton step
- `ut_vector`, `pt_vector`: global finite element vectors, velocity and pressure at previous time step

~~~~
FEShapeVector shape_space_velocity(descr);
FEShapeScalar shape_space_pressure(descr);

// The current nonlinear iterate
FEFunction u(shape_space_velocity, un_vector);
FEFunction p(shape_space_pressure, pn_vector);

// The previous time step
FEFunction u_t(shape_space_velocity, ut_vector);

FETestVector v;
FETestScalar q;

// Set up form language function
// Integrate over all cells
integrate([&] {
  grad (v)*grad(u)
  + v*(grad (u)*u + 1./dt * u_t)
  + div (v)*p + q * div(u);
})
~~~~

## Wishlist

- Gather all terms multiplied with the same test function upon evaluation
- Combine cell and face terms into one form
- Automatic differentiation
- Integration by parts
- LaTeX backend
- Functions (exp, sin, cos,...) of FEFunctions or coordinates
- Cell-wise parameters (mean value, penalty)


~~~~
Current functionality:
- Describe forms in a backend-agnostic way using FEFunction, FETestFunction and similar objects.
- Face forms, cell forms and boundary forms possible.
- Sums and products of FEFunction objects are possible, no common subexpression elimination.
- Backend-speficic data gathered in FEData objects.
- Constants are explcitly strored with the FEFunction objects, no other coefficients currently posssible
- Forms in abstract shape can be transformed into backend-specific shape by a call to `transform`.
- Backend specific integrator objects take the (backend-) Form objects and FEData objects as input and implement a matrix-vector multiplication (or print the form for the LaTex backend) given additional input data like mesh, constraints, etc.
- Sum support for evaluating nonlinear forms available (at least for the MatrixFree backend) by specifying which components to take from the vmult input and which from a previous initialization.
- No automatic differentiation.
