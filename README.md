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

~~~~
FEShapeVector shape_space_velocity(descr);
FEShapeScalar shape_space_pressure(descr);

// The current linear iterate
FEFunction u_l(shape_space_velocity, u_vector);
FEFunction p_l(shape_space_pressure, p_vector);

// The current nonlinear iterate
FEFunction u_n(shape_space_velocity, u_vector);
FEFunction p_n(shape_space_pressure, p_vector);

// The previous time step
FEFunction u_t(shape_space_velocity, u_vector);
FEFunction p_t(shape_space_pressure, p_vector);

FETestVector v;
FETestScalar q;
~~~~