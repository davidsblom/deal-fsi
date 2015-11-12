
/*
 * Author
 *   David Blom, TU Delft. All rights reserved.
 */

DataStorage::DataStorage ()
    :
    degree( 0 ),
    n_global_refines( 0 ),
    final_time( 0 ),
    time_step( 0 ),
    theta( 0 ),
    gravity( 0 ),
    distributed_load( 0 ),
    rho( 0 ),
    E( 0 ),
    nu( 0 ),
    output_paraview( false ),
    output_interval( 0 ),
    prm()
{
    declare_parameters();
}

DataStorage::~DataStorage()
{}

void DataStorage::declare_parameters()
{
    prm.enter_subsection( "Physical constants" );
    {
        prm.declare_entry( "rho", "1000.0", Patterns::Double( 0. ), "Density" );
        prm.declare_entry( "E", "1.4e6", Patterns::Double( 0. ), "Young's modulus" );
        prm.declare_entry( "nu", "0.4", Patterns::Double( 0. ), "Poisson ratio" );
    }
    prm.leave_subsection();

    prm.enter_subsection( "Time step data" );
    {
        prm.declare_entry( "dt", "2.5e-3", Patterns::Double( 0. ), "Time step size" );
        prm.declare_entry( "final_time", "0.05", Patterns::Double( 0. ), "Final time of the simulation" );
        prm.declare_entry( "theta", "1.0", Patterns::Double( 0. ), "The variable theta is used to indicate which time stepping scheme to use, i.e. backward Euler, forward Euler or Crank-Nicolson." );
    }
    prm.leave_subsection();

    prm.enter_subsection( "Space discretization" );
    {
        prm.declare_entry( "fe_degree", "1", Patterns::Integer( 1, 5 ), "The polynomial degree for the displacement space." );
        prm.declare_entry( "n_of_refines", "0", Patterns::Integer( 0, 15 ), "The number of global refines we do on the mesh." );
    }
    prm.leave_subsection();

    prm.enter_subsection( "Boundary conditions" );
    {
        prm.declare_entry( "gravity", "0", Patterns::Double( 0. ), "Gravity in negative y-direction" );
        prm.declare_entry( "traction", "0", Patterns::Double( 0. ), "Traction in negative y-direction" );
    }
    prm.leave_subsection();

    prm.enter_subsection( "Output" );
    {
        prm.declare_entry( "output_interval", "1", dealii::Patterns::Integer( 1 ), "This indicates between how many time steps we print the solution." );
        prm.declare_entry( "output_paraview", "false", dealii::Patterns::Bool(), "This indicates whether the solution is saved in paraview format." );
    }
    prm.leave_subsection();
}

void DataStorage::read_data( const char * filename )
{
    std::ifstream file( filename );

    if ( not file.is_open() )
    {
        std::cout << "Configuration file " << filename << " not found." << std::endl;
        std::cout << "Example of a configuration file:" << std::endl << std::endl;
        prm.print_parameters( std::cout, ParameterHandler::Text );
    }

    AssertThrow( file, ExcFileNotOpen( filename ) );

    prm.read_input( file );

    prm.enter_subsection( "Physical constants" );
    {
        rho = prm.get_double( "rho" );
        E = prm.get_double( "E" );
        nu = prm.get_double( "nu" );
    }
    prm.leave_subsection();

    prm.enter_subsection( "Time step data" );
    {
        time_step = prm.get_double( "dt" );
        final_time = prm.get_double( "final_time" );
        theta = prm.get_double( "theta" );
    }
    prm.leave_subsection();

    prm.enter_subsection( "Space discretization" );
    {
        degree = prm.get_integer( "fe_degree" );
        n_global_refines = prm.get_integer( "n_of_refines" );
    }
    prm.leave_subsection();

    prm.enter_subsection( "Boundary conditions" );
    {
        gravity = prm.get_double( "gravity" );
        distributed_load = prm.get_double( "traction" );
    }
    prm.leave_subsection();

    prm.enter_subsection( "Output" );
    {
        output_interval = prm.get_double( "output_interval" );
        output_paraview = prm.get_bool( "output_paraview" );
    }
    prm.leave_subsection();
}
