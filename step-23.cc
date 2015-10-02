/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2006
 */


// @sect3{Include files}

// We start with the usual assortment of include files that we've seen in so
// many of the previous tests:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

// Here are the only three include files of some new interest: The first one
// is already used, for example, for the
// VectorTools::interpolate_boundary_values and
// MatrixTools::apply_boundary_values functions. However, we here use another
// function in that class, VectorTools::project to compute our initial values
// as the $L^2$ projection of the continuous initial values. Furthermore, we
// use VectorTools::create_right_hand_side to generate the integrals
// $(f^n,\phi^n_i)$. These were previously always generated by hand in
// <code>assemble_system</code> or similar functions in application
// code. However, we're too lazy to do that here, so simply use a library
// function:
#include <deal.II/numerics/vector_tools.h>

// In a very similar vein, we are also too lazy to write the code to assemble
// mass and Laplace matrices, although it would have only taken copying the
// relevant code from any number of previous tutorial programs. Rather, we
// want to focus on the things that are truly new to this program and
// therefore use the MatrixCreator::create_mass_matrix and
// MatrixCreator::create_laplace_matrix functions. They are declared here:
#include <deal.II/numerics/matrix_tools.h>

// Finally, here is an include file that contains all sorts of tool functions
// that one sometimes needs. In particular, we need the
// Utilities::int_to_string class that, given an integer argument, returns a
// string representation of it. It is particularly useful since it allows for
// a second parameter indicating the number of digits to which we want the
// result padded with leading zeros. We will use this to write output files
// that have the form <code>solution-XXX.gnuplot</code> where <code>XXX</code>
// denotes the number of the time step and always consists of three digits
// even if we are still in the single or double digit time steps.
#include <deal.II/base/utilities.h>

#define BOOST_TEST_MODULE SolidMechanicsTest
#include <boost/test/included/unit_test.hpp>

// The last step is as in all previous programs:
namespace Step23
{
  using namespace dealii;


  // @sect3{The <code>WaveEquation</code> class}

  // Next comes the declaration of the main class. It's public interface of
  // functions is like in most of the other tutorial programs. Worth
  // mentioning is that we now have to store four matrices instead of one: the
  // mass matrix $M$, the Laplace matrix $A$, the matrix $M+k^2\theta^2A$ used
  // for solving for $U^n$, and a copy of the mass matrix with boundary
  // conditions applied used for solving for $V^n$. Note that it is a bit
  // wasteful to have an additional copy of the mass matrix around. We will
  // discuss strategies for how to avoid this in the section on possible
  // improvements.
  //
  // Likewise, we need solution vectors for $U^n,V^n$ as well as for the
  // corresponding vectors at the previous time step, $U^{n-1},V^{n-1}$. The
  // <code>system_rhs</code> will be used for whatever right hand side vector
  // we have when solving one of the two linear systems in each time
  // step. These will be solved in the two functions <code>solve_u</code> and
  // <code>solve_v</code>.
  //
  // Finally, the variable <code>theta</code> is used to indicate the
  // parameter $\theta$ that is used to define which time stepping scheme to
  // use, as explained in the introduction. The rest is self-explanatory.
  template <int dim>
  class WaveEquation
  {
  public:
    WaveEquation ( double time_step, double theta );
    void run ();

    void assemble_system();
    void setup_system ();
    void solve_u ();
    void solve_v ();
    void output_results () const;

    Triangulation<dim>   triangulation;
    FESystem<dim>        fe;
    DoFHandler<dim>      dof_handler;

    ConstraintMatrix constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> matrix_u;
    SparseMatrix<double> matrix_v;

    Vector<double>       solution_u, solution_v;
    Vector<double>       old_solution_u, old_solution_v;
    Vector<double>       system_rhs;

    double time, time_step;
    unsigned int timestep_number;
    const double theta;
  };





  template <int dim>
  class RightHandSide :  public Function<dim>
  {
  public:
    RightHandSide ();

    // The next change is that we want a replacement for the
    // <code>value</code> function of the previous examples. There, a second
    // parameter <code>component</code> was given, which denoted which
    // component was requested. Here, we implement a function that returns the
    // whole vector of values at the given place at once, in the second
    // argument of the function. The obvious name for such a replacement
    // function is <code>vector_value</code>.
    //
    // Secondly, in analogy to the <code>value_list</code> function, there is
    // a function <code>vector_value_list</code>, which returns the values of
    // the vector-valued function at several points at once:
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;
  };


  // This is the constructor of the right hand side class. As said above, it
  // only passes down to the base class the number of components, which is
  // <code>dim</code> in the present case (one force component in each of the
  // <code>dim</code> space directions).
  //
  // Some people would have moved the definition of such a short function
  // right into the class declaration. We do not do that, as a matter of
  // style: the deal.II style guides require that class declarations contain
  // only declarations, and that definitions are always to be found
  // outside. This is, obviously, as much as matter of taste as indentation,
  // but we try to be consistent in this direction.
  template <int dim>
  RightHandSide<dim>::RightHandSide ()
    :
    Function<dim> (dim)
  {}


  // Next the function that returns the whole vector of values at the point
  // <code>p</code> at once.
  //
  // To prevent cases where the return vector has not previously been set to
  // the right size we test for this case and otherwise throw an exception at
  // the beginning of the function. Note that enforcing that output arguments
  // already have the correct size is a convention in deal.II, and enforced
  // almost everywhere. The reason is that we would otherwise have to check at
  // the beginning of the function and possibly change the size of the output
  // vector. This is expensive, and would almost always be unnecessary (the
  // first call to the function would set the vector to the right size, and
  // subsequent calls would only have to do redundant checks). In addition,
  // checking and possibly resizing the vector is an operation that can not be
  // removed if we can't rely on the assumption that the vector already has
  // the correct size; this is in contract to the <code>Assert</code> call
  // that is completely removed if the program is compiled in optimized mode.
  //
  // Likewise, if by some accident someone tried to compile and run the
  // program in only one space dimension (in which the elastic equations do
  // not make much sense since they reduce to the ordinary Laplace equation),
  // we terminate the program in the second assertion. The program will work
  // just fine in 3d, however.
  template <int dim>
  inline
  void RightHandSide<dim>::vector_value (const Point<dim> &p,
                                         Vector<double>   &values) const
  {
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));
    Assert (dim >= 2, ExcNotImplemented());

    // The rest of the function implements computing force values. We will use
    // a constant (unit) force in x-direction located in two little circles
    // (or spheres, in 3d) around points (0.5,0) and (-0.5,0), and y-force in
    // an area around the origin; in 3d, the z-component of these centers is
    // zero as well.
    //
    // For this, let us first define two objects that denote the centers of
    // these areas. Note that upon construction of the <code>Point</code>
    // objects, all components are set to zero.
    // Point<dim> point_1, point_2;
    // point_1(0) = 0.5;
    // point_2(0) = -0.5;
    //
    // // If now the point <code>p</code> is in a circle (sphere) of radius 0.2
    // // around one of these points, then set the force in x-direction to one,
    // // otherwise to zero:
    // if (((p-point_1).norm_square() < 0.2*0.2) ||
    //     ((p-point_2).norm_square() < 0.2*0.2))
    //   values(0) = 1;
    // else
    //   values(0) = 0;
    //
    // // Likewise, if <code>p</code> is in the vicinity of the origin, then set
    // // the y-force to 1, otherwise to zero:
    // if (p.norm_square() < 0.2*0.2)
    //   values(1) = 1;
    // else
    //   values(1) = 0;

    double rho = 1000;
    values( 0 ) = 0;
    values( 1 ) = -2.0 * rho;

    double t = this->get_time();
    double T = 0.01;
    double offset = 0.01;

    if ( t - offset < T )
        values( 1 ) *= 0.5 - 0.5 * std::cos( M_PI * (t - offset) / T );

    if ( t < offset )
        values( 1 ) = 0.0;
  }



  // Now, this is the function of the right hand side class that returns the
  // values at several points at once. The function starts out with checking
  // that the number of input and output arguments is equal (the sizes of the
  // individual output vectors will be checked in the function that we call
  // further down below). Next, we define an abbreviation for the number of
  // points which we shall work on, to make some things simpler below.
  template <int dim>
  void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                              std::vector<Vector<double> >   &value_list) const
  {
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    // Finally we treat each of the points. In one of the previous examples,
    // we have explained why the
    // <code>value_list</code>/<code>vector_value_list</code> function had
    // been introduced: to prevent us from calling virtual functions too
    // frequently. On the other hand, we now need to implement the same
    // function twice, which can lead to confusion if one function is changed
    // but the other is not.
    //
    // We can prevent this situation by calling
    // <code>RightHandSide::vector_value</code> on each point in the input
    // list. Note that by giving the full name of the function, including the
    // class name, we instruct the compiler to explicitly call this function,
    // and not to use the virtual function call mechanism that would be used
    // if we had just called <code>vector_value</code>. This is important,
    // since the compiler generally can't make any assumptions which function
    // is called when using virtual functions, and it therefore can't inline
    // the called function into the site of the call. On the contrary, here we
    // give the fully qualified name, which bypasses the virtual function
    // call, and consequently the compiler knows exactly which function is
    // called and will inline above function into the present location. (Note
    // that we have declared the <code>vector_value</code> function above
    // <code>inline</code>, though modern compilers are also able to inline
    // functions even if they have not been declared as inline).
    //
    // It is worth noting why we go to such length explaining what we
    // do. Using this construct, we manage to avoid any inconsistency: if we
    // want to change the right hand side function, it would be difficult to
    // always remember that we always have to change two functions in the same
    // way. Using this forwarding mechanism, we only have to change a single
    // place (the <code>vector_value</code> function), and the second place
    // (the <code>vector_value_list</code> function) will always be consistent
    // with it. At the same time, using virtual function call bypassing, the
    // code is no less efficient than if we had written it twice in the first
    // place:
    for (unsigned int p=0; p<n_points; ++p)
      RightHandSide<dim>::vector_value (points[p],
                                        value_list[p]);
  }



  // @sect3{Implementation of the <code>WaveEquation</code> class}

  // The implementation of the actual logic is actually fairly short, since we
  // relegate things like assembling the matrices and right hand side vectors
  // to the library. The rest boils down to not much more than 130 lines of
  // actual code, a significant fraction of which is boilerplate code that can
  // be taken from previous example programs (e.g. the functions that solve
  // linear systems, or that generate output).
  //
  // Let's start with the constructor (for an explanation of the choice of
  // time step, see the section on Courant, Friedrichs, and Lewy in the
  // introduction):
  template <int dim>
  WaveEquation<dim>::WaveEquation ( double time_step, double theta ) :
    fe (FE_Q<dim>(1), dim),
    dof_handler (triangulation),
    time_step( time_step ),
    theta( theta )
    // time_step (1./64),
    // theta (0.5)
  {}


  // @sect4{WaveEquation::setup_system}

  // The next function is the one that sets up the mesh, DoFHandler, and
  // matrices and vectors at the beginning of the program, i.e. before the
  // first time step. The first few lines are pretty much standard if you've
  // read through the tutorial programs at least up to step-6:
  template <int dim>
  void WaveEquation<dim>::setup_system ()
  {
    // GridGenerator::hyper_cube (triangulation, -1, 1);

    double x1 = 0.24899;
    double y1 = 0.21;
    double x2 = 0.6;
    double y2 = 0.19;

    dealii::Point<dim, double> point1( x1, y1 );
    dealii::Point<dim, double> point2( x2, y2 );

    std::vector<unsigned int> repetitions( dim );
    repetitions[0] = 35;
    repetitions[1] = 2;

    // GridGenerator::hyper_rectangle( triangulation, point1, point2 );
    dealii::GridGenerator::subdivided_hyper_rectangle( triangulation, repetitions, point1, point2, true );

    triangulation.refine_global (2);

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs (fe);

    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from (dsp);

    // Then comes a block where we have to initialize the 3 matrices we need
    // in the course of the program: the mass matrix, the Laplace matrix, and
    // the matrix $M+k^2\theta^2A$ used when solving for $U^n$ in each time
    // step.
    //
    // When setting up these matrices, note that they all make use of the same
    // sparsity pattern object. Finally, the reason why matrices and sparsity
    // patterns are separate objects in deal.II (unlike in many other finite
    // element or linear algebra classes) becomes clear: in a significant
    // fraction of applications, one has to hold several matrices that happen
    // to have the same sparsity pattern, and there is no reason for them not
    // to share this information, rather than re-building and wasting memory
    // on it several times.
    //
    // After initializing all of these matrices, we call library functions
    // that build the Laplace and mass matrices. All they need is a DoFHandler
    // object and a quadrature formula object that is to be used for numerical
    // integration. Note that in many respects these functions are better than
    // what we would usually do in application programs, for example because
    // they automatically parallelize building the matrices if multiple
    // processors are available in a machine. The matrices for solving linear
    // systems will be filled in the run() method because we need to re-apply
    // boundary conditions every time step.
    mass_matrix.reinit (sparsity_pattern);
    laplace_matrix.reinit (sparsity_pattern);
    matrix_u.reinit (sparsity_pattern);
    matrix_v.reinit (sparsity_pattern);

    MatrixCreator::create_mass_matrix (dof_handler, QGauss<dim>(3),
                                       mass_matrix);
    //MatrixCreator::create_laplace_matrix (dof_handler, QGauss<dim>(3),
    //                                      laplace_matrix);


    // The rest of the function is spent on setting vector sizes to the
    // correct value. The final line closes the hanging node constraints
    // object. Since we work on a uniformly refined mesh, no constraints exist
    // or have been computed (i.e. there was no need to call
    // DoFTools::make_hanging_node_constraints as in other programs), but we
    // need a constraints object in one place further down below anyway.
    solution_u.reinit (dof_handler.n_dofs());
    solution_v.reinit (dof_handler.n_dofs());
    old_solution_u.reinit (dof_handler.n_dofs());
    old_solution_v.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());

    constraints.close ();

    assemble_system();
  }

  template <int dim>
  void WaveEquation<dim>::assemble_system ()
  {
      QGauss<dim>  quadrature_formula(2);

      FEValues<dim> fe_values (fe, quadrature_formula,
                               update_values   | update_gradients |
                               update_quadrature_points | update_JxW_values);

      const unsigned int   dofs_per_cell = fe.dofs_per_cell;
      const unsigned int   n_q_points    = quadrature_formula.size();

      FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      // As was shown in previous examples as well, we need a place where to
      // store the values of the coefficients at all the quadrature points on a
      // cell. In the present situation, we have two coefficients, lambda and
      // mu.
      std::vector<double>     lambda_values (n_q_points);
      std::vector<double>     mu_values (n_q_points);

      // Well, we could as well have omitted the above two arrays since we will
      // use constant coefficients for both lambda and mu, which can be declared
      // like this. They both represent functions always returning the constant
      // value 1.0. Although we could omit the respective factors in the
      // assemblage of the matrix, we use them here for purpose of
      // demonstration.
    //   ConstantFunction<dim> lambda(1.), mu(1.);

      double nu = 0.4;
      double E = 1.4e6;
      double mu_s = E / ( 2.0 * ( 1.0 + nu ) );
      double lambda_s = nu * E / ( (1.0 + nu)*(1.0 - 2.0*nu) );

      ConstantFunction<dim> lambda(lambda_s), mu(mu_s);

      // Now we can begin with the loop over all cells:
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                     endc = dof_handler.end();
      for (; cell!=endc; ++cell)
        {
          cell_matrix = 0;

          fe_values.reinit (cell);

          // Next we get the values of the coefficients at the quadrature
          // points. Likewise for the right hand side:
          lambda.value_list (fe_values.get_quadrature_points(), lambda_values);
          mu.value_list     (fe_values.get_quadrature_points(), mu_values);

          // Then assemble the entries of the local stiffness matrix and right
          // hand side vector. This follows almost one-to-one the pattern
          // described in the introduction of this example.  One of the few
          // comments in place is that we can compute the number
          // <code>comp(i)</code>, i.e. the index of the only nonzero vector
          // component of shape function <code>i</code> using the
          // <code>fe.system_to_component_index(i).first</code> function call
          // below.
          //
          // (By accessing the <code>first</code> variable of the return value
          // of the <code>system_to_component_index</code> function, you might
          // already have guessed that there is more in it. In fact, the
          // function returns a <code>std::pair@<unsigned int, unsigned
          // int@></code>, of which the first element is <code>comp(i)</code>
          // and the second is the value <code>base(i)</code> also noted in the
          // introduction, i.e.  the index of this shape function within all the
          // shape functions that are nonzero in this component,
          // i.e. <code>base(i)</code> in the diction of the introduction. This
          // is not a number that we are usually interested in, however.)
          //
          // With this knowledge, we can assemble the local matrix
          // contributions:
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int
              component_i = fe.system_to_component_index(i).first;

              for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                  const unsigned int
                  component_j = fe.system_to_component_index(j).first;

                  for (unsigned int q_point=0; q_point<n_q_points;
                       ++q_point)
                    {
                      cell_matrix(i,j)
                      +=
                        // The first term is (lambda d_i u_i, d_j v_j) + (mu d_i
                        // u_j, d_j v_i).  Note that
                        // <code>shape_grad(i,q_point)</code> returns the
                        // gradient of the only nonzero component of the i-th
                        // shape function at quadrature point q_point. The
                        // component <code>comp(i)</code> of the gradient, which
                        // is the derivative of this only nonzero vector
                        // component of the i-th shape function with respect to
                        // the comp(i)th coordinate is accessed by the appended
                        // brackets.
                        (
                          (fe_values.shape_grad(i,q_point)[component_i] *
                           fe_values.shape_grad(j,q_point)[component_j] *
                           lambda_values[q_point])
                          +
                          (fe_values.shape_grad(i,q_point)[component_j] *
                           fe_values.shape_grad(j,q_point)[component_i] *
                           mu_values[q_point])
                          +
                          // The second term is (mu nabla u_i, nabla v_j).  We
                          // need not access a specific component of the
                          // gradient, since we only have to compute the scalar
                          // product of the two gradients, of which an
                          // overloaded version of the operator* takes care, as
                          // in previous examples.
                          //
                          // Note that by using the ?: operator, we only do this
                          // if comp(i) equals comp(j), otherwise a zero is
                          // added (which will be optimized away by the
                          // compiler).
                          ((component_i == component_j) ?
                           (fe_values.shape_grad(i,q_point) *
                            fe_values.shape_grad(j,q_point) *
                            mu_values[q_point])  :
                           0)
                        )
                        *
                        fe_values.JxW(q_point);
                    }
                }
            }


          // The transfer from local degrees of freedom into the global matrix
          // and right hand side vector does not depend on the equation under
          // consideration, and is thus the same as in all previous
          // examples. The same holds for the elimination of hanging nodes from
          // the matrix and right hand side, once we are done with assembling
          // the entire linear system:
          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                laplace_matrix.add (local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i,j));

            }
        }

      constraints.condense (laplace_matrix);
  }


  // @sect4{WaveEquation::solve_u and WaveEquation::solve_v}

  // The next two functions deal with solving the linear systems associated
  // with the equations for $U^n$ and $V^n$. Both are not particularly
  // interesting as they pretty much follow the scheme used in all the
  // previous tutorial programs.
  //
  // One can make little experiments with preconditioners for the two matrices
  // we have to invert. As it turns out, however, for the matrices at hand
  // here, using Jacobi or SSOR preconditioners reduces the number of
  // iterations necessary to solve the linear system slightly, but due to the
  // cost of applying the preconditioner it is no win in terms of run-time. It
  // is not much of a loss either, but let's keep it simple and just do
  // without:
  template <int dim>
  void WaveEquation<dim>::solve_u ()
  {
    SolverControl           solver_control (1000, 1e-12*system_rhs.l2_norm());
    SolverCG<>              cg (solver_control);

    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(matrix_u);

    // cg.solve (matrix_u, solution_u, system_rhs,
    //           preconditioner);

    A_direct.vmult (solution_u, system_rhs);

    std::cout << "   u-equation: " << solver_control.last_step()
              << " CG iterations."
              << std::endl;
  }


  template <int dim>
  void WaveEquation<dim>::solve_v ()
  {
    SolverControl           solver_control (1000, 1e-12*system_rhs.l2_norm());
    SolverCG<>              cg (solver_control);

    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(matrix_v);

    // cg.solve (matrix_v, solution_v, system_rhs,
    //           preconditioner);

    A_direct.vmult (solution_v, system_rhs);

    std::cout << "   v-equation: " << solver_control.last_step()
              << " CG iterations."
              << std::endl;
  }



  // @sect4{WaveEquation::output_results}

  // Likewise, the following function is pretty much what we've done
  // before. The only thing worth mentioning is how here we generate a string
  // representation of the time step number padded with leading zeros to 3
  // character length using the Utilities::int_to_string function's second
  // argument.
  template <int dim>
  void WaveEquation<dim>::output_results () const
  {
      return;
    DataOut<dim> data_out;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
  data_component_interpretation(dim,
                                DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_name_u(dim, "displacement");
    std::vector<std::string> solution_name_v(dim, "velocity");

    data_out.attach_dof_handler (dof_handler);

    data_out.add_data_vector(solution_u,
                           solution_name_u,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
    data_out.add_data_vector(solution_v,
                          solution_name_v,
                          DataOut<dim>::type_dof_data,
                          data_component_interpretation);

    data_out.build_patches ();

    const std::string filename = "solution-" +
                                 Utilities::int_to_string (timestep_number, 3) +
                                 ".vtk";
    std::ofstream output (filename.c_str());
    data_out.write_vtk (output);
  }




  // @sect4{WaveEquation::run}

  // The following is really the only interesting function of the program. It
  // contains the loop over all time steps, but before we get to that we have
  // to set up the grid, DoFHandler, and matrices. In addition, we have to
  // somehow get started with initial values. To this end, we use the
  // VectorTools::project function that takes an object that describes a
  // continuous function and computes the $L^2$ projection of this function
  // onto the finite element space described by the DoFHandler object. Can't
  // be any simpler than that:
  template <int dim>
  void WaveEquation<dim>::run ()
  {
    setup_system();

    // The next thing is to loop over all the time steps until we reach the
    // end time ($T=5$ in this case). In each time step, we first have to
    // solve for $U^n$, using the equation $(M^n + k^2\theta^2 A^n)U^n =$
    // $(M^{n,n-1} - k^2\theta(1-\theta) A^{n,n-1})U^{n-1} + kM^{n,n-1}V^{n-1}
    // +$ $k\theta \left[k \theta F^n + k(1-\theta) F^{n-1} \right]$. Note
    // that we use the same mesh for all time steps, so that $M^n=M^{n,n-1}=M$
    // and $A^n=A^{n,n-1}=A$. What we therefore have to do first is to add up
    // $MU^{n-1} - k^2\theta(1-\theta) AU^{n-1} + kMV^{n-1}$ and the forcing
    // terms, and put the result into the <code>system_rhs</code> vector. (For
    // these additions, we need a temporary vector that we declare before the
    // loop to avoid repeated memory allocations in each time step.)
    //
    // The one thing to realize here is how we communicate the time variable
    // to the object describing the right hand side: each object derived from
    // the Function class has a time field that can be set using the
    // Function::set_time and read by Function::get_time. In essence, using
    // this mechanism, all functions of space and time are therefore
    // considered functions of space evaluated at a particular time. This
    // matches well what we typically need in finite element programs, where
    // we almost always work on a single time step at a time, and where it
    // never happens that, for example, one would like to evaluate a
    // space-time function for all times at any given spatial location.
    Vector<double> tmp (solution_u.size());
    Vector<double> forcing_terms (solution_u.size());

    double initial_time = 0;
    double final_time = 0.05;

    double rho = 1000.0;

    timestep_number = 0;

    output_results();

    timestep_number = 1;
    time = initial_time + time_step;

    while ( time <= final_time )

    // for (timestep_number=1, time=time_step;
    //      time<=0.5;
    //      time+=time_step, ++timestep_number)
      {
        std::cout << "Time step " << timestep_number
                  << " at t=" << time
                  << std::endl;

        mass_matrix.vmult (system_rhs, old_solution_u);

        mass_matrix.vmult (tmp, old_solution_v);
        system_rhs.add (time_step, tmp);

        laplace_matrix.vmult (tmp, old_solution_u);
        system_rhs.add (-theta * (1-theta) * time_step * time_step / rho, tmp);

        RightHandSide<dim> rhs_function;
        rhs_function.set_time (time);
        VectorTools::create_right_hand_side (dof_handler, QGauss<dim>(2),
                                             rhs_function, tmp);
        forcing_terms = tmp;
        forcing_terms *= theta * time_step;

        rhs_function.set_time (time-time_step);
        VectorTools::create_right_hand_side (dof_handler, QGauss<dim>(2),
                                             rhs_function, tmp);

        forcing_terms.add ((1-theta) * time_step, tmp);
        forcing_terms *= 1.0 / rho;

        system_rhs.add (theta * time_step, forcing_terms);

        // After so constructing the right hand side vector of the first
        // equation, all we have to do is apply the correct boundary
        // values. As for the right hand side, this is a space-time function
        // evaluated at a particular time, which we interpolate at boundary
        // nodes and then use the result to apply boundary values as we
        // usually do. The result is then handed off to the solve_u()
        // function:
        {
            std::map<types::global_dof_index,double> boundary_values;
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      0,
                                                      ZeroFunction<dim>(dim),
                                                      boundary_values);


          // The matrix for solve_u() is the same in every time steps, so one
          // could think that it is enough to do this only once at the
          // beginning of the simulation. However, since we need to apply
          // boundary values to the linear system (which eliminate some matrix
          // rows and columns and give contributions to the right hand side),
          // we have to refill the matrix in every time steps before we
          // actually apply boundary data. The actual content is very simple:
          // it is the sum of the mass matrix and a weighted Laplace matrix:
          matrix_u.copy_from (mass_matrix);
          matrix_u.add (theta * theta * time_step * time_step / rho, laplace_matrix);
          MatrixTools::apply_boundary_values (boundary_values,
                                              matrix_u,
                                              solution_u,
                                              system_rhs);
        }
        solve_u ();


        // The second step, i.e. solving for $V^n$, works similarly, except
        // that this time the matrix on the left is the mass matrix (which we
        // copy again in order to be able to apply boundary conditions, and
        // the right hand side is $MV^{n-1} - k\left[ \theta A U^n +
        // (1-\theta) AU^{n-1}\right]$ plus forcing terms. %Boundary values
        // are applied in the same way as before, except that now we have to
        // use the BoundaryValuesV class:
        laplace_matrix.vmult (system_rhs, solution_u);
        system_rhs *= -theta * time_step / rho;

        mass_matrix.vmult (tmp, old_solution_v);
        system_rhs += tmp;

        laplace_matrix.vmult (tmp, old_solution_u);
        system_rhs.add (-time_step * (1-theta) / rho, tmp);

        system_rhs += forcing_terms;

        {
            std::map<types::global_dof_index,double> boundary_values;
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      0,
                                                      ZeroFunction<dim>(dim),
                                                      boundary_values);
          matrix_v.copy_from (mass_matrix);
          MatrixTools::apply_boundary_values (boundary_values,
                                              matrix_v,
                                              solution_v,
                                              system_rhs);
        }
        solve_v ();

        // Finally, after both solution components have been computed, we
        // output the result, compute the energy in the solution, and go on to
        // the next time step after shifting the present solution into the
        // vectors that hold the solution at the previous time step. Note the
        // function SparseMatrix::matrix_norm_square that can compute
        // $\left<V^n,MV^n\right>$ and $\left<U^n,AU^n\right>$ in one step,
        // saving us the expense of a temporary vector and several lines of
        // code:
        output_results ();

        old_solution_u = solution_u;
        old_solution_v = solution_v;

        timestep_number++;
        time = initial_time + timestep_number * time_step;
      }

      timestep_number--;
      time = initial_time + timestep_number * time_step;
  }
}


// @sect3{The <code>main</code> function}

// What remains is the main function of the program. There is nothing here
// that hasn't been shown in several of the previous programs:


BOOST_AUTO_TEST_CASE( crank_nicolson_test )
{
  using namespace dealii;
  using namespace Step23;

  deallog.depth_console (0);

  double time_step = 2.5e-3;
  double theta = 0.5;

  unsigned int nbComputations = 4;

  std::vector<unsigned int> nbTimeSteps( nbComputations );
  std::vector<double> solution_l2_norm( nbComputations );

  for ( unsigned int i = 0; i < nbComputations; ++i )
  {
      double dt = time_step / std::pow( 2, i );

      WaveEquation<2> wave_equation_solver ( dt, theta );
      wave_equation_solver.run ();

      if ( i > 0 )
          assert( nbTimeSteps[i - 1] * 2 == wave_equation_solver.timestep_number );

      double l2norm = 0;
      for ( unsigned int i = 0; i < wave_equation_solver.solution_v.size(); ++i )
        l2norm += wave_equation_solver.solution_v[i] * wave_equation_solver.solution_v[i];
      l2norm = std::sqrt( l2norm );

      solution_l2_norm[i] = l2norm;
      nbTimeSteps[i] = wave_equation_solver.timestep_number;
  }

  std::vector<double> error( nbComputations - 1 );

  for ( unsigned int i = 0; i < error.size(); ++i )
      error[i] = std::abs( solution_l2_norm[i] - solution_l2_norm[nbComputations - 1] ) / std::abs( solution_l2_norm[nbComputations - 1] );

  std::vector<double> order( nbComputations - 2 );

  for ( unsigned int i = 0; i < order.size(); ++i )
  {
      double dti = time_step / std::pow( 2, i );
      double dtinew = time_step / std::pow( 2, i + 1 );
      order[i] = std::log10( error[i + 1] ) - std::log10( error[i] );
      order[i] /= std::log10( dtinew ) - std::log10( dti );

      BOOST_CHECK_GE( order[i], 2 );
  }

}
