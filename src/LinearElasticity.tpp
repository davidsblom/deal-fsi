
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
WaveEquation<dim>::WaveEquation (
    double time_step,
    double theta,
    double degree,
    double gravity,
    double distributed_load,
    unsigned int n_global_refines
    ) :
    deg( degree ),
    n_global_refines( n_global_refines ),
    fe( FE_Q<dim>(degree), dim ),
    dof_handler( triangulation ),
    time_step( time_step ),
    theta( theta ),
    gravity( gravity ),
    distributed_load( distributed_load ),
    init( false )
{
    assert( degree >= 1 );
    assert( time_step > 0 );
    assert( theta >= 0 && theta <= 1 );
}

// @sect4{WaveEquation::setup_system}

// The next function is the one that sets up the mesh, DoFHandler, and
// matrices and vectors at the beginning of the program, i.e. before the
// first time step. The first few lines are pretty much standard if you've
// read through the tutorial programs at least up to step-6:
template <int dim>
void WaveEquation<dim>::setup_system()
{
    // GridGenerator::hyper_cube (triangulation, -1, 1);

    double x1 = 0.24899;
    double y1 = 0.21;
    double x2 = 0.6;
    double y2 = 0.19;

    Point<dim, double> point1( x1, y1 );
    Point<dim, double> point2( x2, y2 );

    std::vector<unsigned int> repetitions( dim );
    repetitions[0] = 35;
    repetitions[1] = 2;

    // GridGenerator::hyper_rectangle( triangulation, point1, point2 );
    GridGenerator::subdivided_hyper_rectangle( triangulation, repetitions, point1, point2, true );

    triangulation.refine_global( n_global_refines );

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs( fe );

    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    DynamicSparsityPattern dsp( dof_handler.n_dofs(), dof_handler.n_dofs() );
    DoFTools::make_sparsity_pattern( dof_handler, dsp );
    sparsity_pattern.copy_from( dsp );

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
    mass_matrix.reinit( sparsity_pattern );
    laplace_matrix.reinit( sparsity_pattern );
    matrix_u.reinit( sparsity_pattern );
    matrix_v.reinit( sparsity_pattern );

    MatrixCreator::create_mass_matrix( dof_handler, QGauss<dim>( 3 ),
        mass_matrix );

    // MatrixCreator::create_laplace_matrix (dof_handler, QGauss<dim>(3),
    // laplace_matrix);


    // The rest of the function is spent on setting vector sizes to the
    // correct value. The final line closes the hanging node constraints
    // object. Since we work on a uniformly refined mesh, no constraints exist
    // or have been computed (i.e. there was no need to call
    // DoFTools::make_hanging_node_constraints as in other programs), but we
    // need a constraints object in one place further down below anyway.
    solution_u.reinit( dof_handler.n_dofs() );
    solution_v.reinit( dof_handler.n_dofs() );
    old_solution_u.reinit( dof_handler.n_dofs() );
    old_solution_v.reinit( dof_handler.n_dofs() );
    system_rhs.reinit( dof_handler.n_dofs() );
    body_force.reinit( dof_handler.n_dofs() );
    old_body_force.reinit( dof_handler.n_dofs() );

    constraints.close();

    assemble_system();
}

template <int dim>
void WaveEquation<dim>::assemble_system()
{
    body_force.reinit( dof_handler.n_dofs() );
    laplace_matrix.reinit( sparsity_pattern );

    QGauss<dim>  quadrature_formula( deg + 1 );

    FEValues<dim> fe_values( fe, quadrature_formula,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values );

    QGauss<dim - 1>  quadrature_formula_face( deg + 1 );
    FEFaceValues<dim> fe_face_values( fe, quadrature_formula_face, update_values | update_quadrature_points | update_JxW_values );

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double>   cell_matrix( dofs_per_cell, dofs_per_cell );
    Vector<double> cell_rhs( dofs_per_cell );

    std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );

    unsigned int dofs_per_face = fe.n_dofs_per_face();

    // As was shown in previous examples as well, we need a place where to
    // store the values of the coefficients at all the quadrature points on a
    // cell. In the present situation, we have two coefficients, lambda and
    // mu.
    std::vector<double>     lambda_values( n_q_points );
    std::vector<double>     mu_values( n_q_points );

    // Well, we could as well have omitted the above two arrays since we will
    // use constant coefficients for both lambda and mu, which can be declared
    // like this. They both represent functions always returning the constant
    // value 1.0. Although we could omit the respective factors in the
    // assemblage of the matrix, we use them here for purpose of
    // demonstration.
    // ConstantFunction<dim> lambda(1.), mu(1.);

    double nu = 0.4;
    double E = 1.4e6;
    double mu_s = E / ( 2.0 * (1.0 + nu) );
    double lambda_s = nu * E / ( (1.0 + nu) * (1.0 - 2.0 * nu) );

    ConstantFunction<dim> lambda( lambda_s ), mu( mu_s );

    // Now we can begin with the loop over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    for (; cell != endc; ++cell )
    {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit( cell );

        // Next we get the values of the coefficients at the quadrature
        // points. Likewise for the right hand side:
        lambda.value_list( fe_values.get_quadrature_points(), lambda_values );
        mu.value_list( fe_values.get_quadrature_points(), mu_values );

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
        for ( unsigned int i = 0; i < dofs_per_cell; ++i )
        {
            const unsigned int
                component_i = fe.system_to_component_index( i ).first;

            for ( unsigned int j = 0; j < dofs_per_cell; ++j )
            {
                const unsigned int
                    component_j = fe.system_to_component_index( j ).first;

                for ( unsigned int q_point = 0; q_point < n_q_points;
                    ++q_point )
                {
                    cell_matrix( i, j )
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
                        (fe_values.shape_grad( i, q_point )[component_i] *
                            fe_values.shape_grad( j, q_point )[component_j] *
                            lambda_values[q_point])
                        +
                        (fe_values.shape_grad( i, q_point )[component_j] *
                            fe_values.shape_grad( j, q_point )[component_i] *
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
                        ( (component_i == component_j) ?
                            (fe_values.shape_grad( i, q_point ) *
                                fe_values.shape_grad( j, q_point ) *
                                mu_values[q_point])  :
                            0 )
                        )
                        *
                        fe_values.JxW( q_point );
                }
            }
        }

        for ( unsigned int i = 0; i < dofs_per_cell; ++i )
        {
            const unsigned int component_i = fe.system_to_component_index( i ).first;

            for ( unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face )
            {
                if ( cell->face( face )->at_boundary() == true
                    && cell->face( face )->boundary_id() == 3 )
                {
                    fe_face_values.reinit( cell, face );

                    std::vector<unsigned int> local_face_dof_indices( dofs_per_face );
                    cell->face( face )->get_dof_indices( local_face_dof_indices );

                    assert( dofs_per_face == fe_face_values.n_quadrature_points * dim );

                    for ( unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q )
                        cell_rhs( i ) += -get_traction( component_i, local_face_dof_indices[q * dim + component_i] ) * fe_face_values.shape_value( i, q ) * fe_face_values.JxW( q );
                }
            }
        }

        // The transfer from local degrees of freedom into the global matrix
        // and right hand side vector does not depend on the equation under
        // consideration, and is thus the same as in all previous
        // examples. The same holds for the elimination of hanging nodes from
        // the matrix and right hand side, once we are done with assembling
        // the entire linear system:
        cell->get_dof_indices( local_dof_indices );

        for ( unsigned int i = 0; i < dofs_per_cell; ++i )
        {
            for ( unsigned int j = 0; j < dofs_per_cell; ++j )
                laplace_matrix.add( local_dof_indices[i],
                    local_dof_indices[j],
                    cell_matrix( i, j ) );

            body_force( local_dof_indices[i] ) += cell_rhs( i );
        }
    }

    constraints.condense( laplace_matrix );
}

template <int dim>
void WaveEquation<dim>::initTimeStep()
{
    assert( !init );

    std::cout << "Time step " << timestep_number
              << " at t=" << time
              << std::endl;

    init = true;
}

template <int dim>
void WaveEquation<dim>::finalizeTimeStep()
{
    assert( init );

    output_results();

    old_solution_u = solution_u;
    old_solution_v = solution_v;
    old_body_force = body_force;

    timestep_number++;
    time = initial_time + timestep_number * time_step;

    init = false;
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
void WaveEquation<dim>::solve_u()
{
    SolverControl solver_control( 1000, 1e-12 * system_rhs.l2_norm() );
    SolverCG<>              cg( solver_control );

    SparseDirectUMFPACK A_direct;
    A_direct.initialize( matrix_u );

    // cg.solve (matrix_u, solution_u, system_rhs,
    // preconditioner);

    A_direct.vmult( solution_u, system_rhs );

    std::cout << "   u-equation: " << solver_control.last_step()
              << " CG iterations."
              << std::endl;
}

template <int dim>
void WaveEquation<dim>::solve_v()
{
    SolverControl solver_control( 1000, 1e-12 * system_rhs.l2_norm() );
    SolverCG<>              cg( solver_control );

    SparseDirectUMFPACK A_direct;
    A_direct.initialize( matrix_v );

    // cg.solve (matrix_v, solution_v, system_rhs,
    // preconditioner);

    A_direct.vmult( solution_v, system_rhs );

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
void WaveEquation<dim>::output_results() const
{
    return;
    DataOut<dim> data_out;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation( dim,
        DataComponentInterpretation::component_is_part_of_vector );
    std::vector<std::string> solution_name_u( dim, "displacement" );
    std::vector<std::string> solution_name_v( dim, "velocity" );

    data_out.attach_dof_handler( dof_handler );

    data_out.add_data_vector( solution_u,
        solution_name_u,
        DataOut<dim>::type_dof_data,
        data_component_interpretation );
    data_out.add_data_vector( solution_v,
        solution_name_v,
        DataOut<dim>::type_dof_data,
        data_component_interpretation );

    data_out.build_patches();

    const std::string filename = "solution-" +
        Utilities::int_to_string( timestep_number, 3 ) +
        ".vtk";
    std::ofstream output( filename.c_str() );
    data_out.write_vtk( output );
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
void WaveEquation<dim>::run()
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
    Vector<double> tmp( solution_u.size() );
    Vector<double> forcing_terms( solution_u.size() );

    double initial_time = 0;
    double final_time = 0.05;

    double rho = 1000.0;

    timestep_number = 0;

    output_results();

    timestep_number = 1;
    time = initial_time + time_step;

    while ( time <= final_time )
    {
        initTimeStep();

        assemble_system();

        // mass_matrix.vmult (system_rhs, old_solution_u);
        system_rhs = mass_matrix * old_solution_u;

        // mass_matrix.vmult (tmp, old_solution_v);
        tmp = mass_matrix * old_solution_v;
        system_rhs.add( time_step, tmp );

        // laplace_matrix.vmult (tmp, old_solution_u);
        tmp = laplace_matrix * old_solution_u;
        system_rhs.add( -theta * (1 - theta) * time_step * time_step / rho, tmp );

        RightHandSide<dim> rhs_function( gravity );
        rhs_function.set_time( time );
        VectorTools::create_right_hand_side( dof_handler, QGauss<dim>( 2 ),
            rhs_function, tmp );
        tmp += body_force;
        forcing_terms = tmp;
        forcing_terms *= theta * time_step;

        rhs_function.set_time( time - time_step );
        VectorTools::create_right_hand_side( dof_handler, QGauss<dim>( 2 ),
            rhs_function, tmp );

        tmp += old_body_force;
        forcing_terms.add( (1 - theta) * time_step, tmp );
        forcing_terms *= 1.0 / rho;

        system_rhs.add( theta * time_step, forcing_terms );

        // After so constructing the right hand side vector of the first
        // equation, all we have to do is apply the correct boundary
        // values. As for the right hand side, this is a space-time function
        // evaluated at a particular time, which we interpolate at boundary
        // nodes and then use the result to apply boundary values as we
        // usually do. The result is then handed off to the solve_u()
        // function:
        {
            std::map<types::global_dof_index, double> boundary_values;
            VectorTools::interpolate_boundary_values( dof_handler,
                0,
                ZeroFunction<dim>( dim ),
                boundary_values );


            // The matrix for solve_u() is the same in every time steps, so one
            // could think that it is enough to do this only once at the
            // beginning of the simulation. However, since we need to apply
            // boundary values to the linear system (which eliminate some matrix
            // rows and columns and give contributions to the right hand side),
            // we have to refill the matrix in every time steps before we
            // actually apply boundary data. The actual content is very simple:
            // it is the sum of the mass matrix and a weighted Laplace matrix:
            matrix_u.copy_from( mass_matrix );
            matrix_u.add( theta * theta * time_step * time_step / rho, laplace_matrix );
            MatrixTools::apply_boundary_values( boundary_values,
                matrix_u,
                solution_u,
                system_rhs );
        }
        solve_u();


        // The second step, i.e. solving for $V^n$, works similarly, except
        // that this time the matrix on the left is the mass matrix (which we
        // copy again in order to be able to apply boundary conditions, and
        // the right hand side is $MV^{n-1} - k\left[ \theta A U^n +
        // (1-\theta) AU^{n-1}\right]$ plus forcing terms. %Boundary values
        // are applied in the same way as before, except that now we have to
        // use the BoundaryValuesV class:
        // laplace_matrix.vmult (system_rhs, solution_u);
        system_rhs = laplace_matrix * solution_u;
        system_rhs *= -theta * time_step / rho;

        // mass_matrix.vmult (tmp, old_solution_v);
        tmp = mass_matrix * old_solution_v;
        system_rhs += tmp;

        // laplace_matrix.vmult (tmp, old_solution_u);
        tmp = laplace_matrix * old_solution_u;
        system_rhs.add( -time_step * (1 - theta) / rho, tmp );

        system_rhs += forcing_terms;

        {
            std::map<types::global_dof_index, double> boundary_values;
            VectorTools::interpolate_boundary_values( dof_handler,
                0,
                ZeroFunction<dim>( dim ),
                boundary_values );
            matrix_v.copy_from( mass_matrix );
            MatrixTools::apply_boundary_values( boundary_values,
                matrix_v,
                solution_v,
                system_rhs );
        }
        solve_v();

        finalizeTimeStep();
    }

    timestep_number--;
    time = initial_time + timestep_number * time_step;
}

template <int dim>
unsigned int WaveEquation<dim>::n_dofs() const
{
    return dof_handler.n_dofs();
}

template <int dim>
double WaveEquation<dim>::point_value() const
{
    Point<dim> point( 0.6, 0.2 );

    Vector<double> vector_value( dim );

    VectorTools::point_value( dof_handler,
        solution_u,
        point,
        vector_value
        );

    std::cout << "   Point value = " << vector_value[1] << std::endl;

    return vector_value[1];
}

template <int dim>
double WaveEquation<dim>::get_traction(
    const unsigned int component_i,
    const unsigned int
    )
{
    // return 0;
    double t = time;
    double T = 0.01;
    double offset = 0.01;
    double value = distributed_load;

    if ( t - offset < T )
        value *= 0.5 - 0.5 * std::cos( M_PI * (t - offset) / T );

    if ( t < offset )
        value = 0.0;

    if ( component_i == 1 )
        return value;

    return 0.0;
}

template <class Scalar>
Vector<Scalar> operator *(
    SparseMatrix<Scalar> & A,
    Vector<Scalar> & b
    )
{
    Vector<Scalar> tmp( b.size() );
    A.vmult( tmp, b );
    return tmp;
}
