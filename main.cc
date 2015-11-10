
/*
 * Author
 *   David Blom, TU Delft. All rights reserved.
 */

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
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/utilities.h>

#include "include/RightHandSide.h"
#include "include/LinearElasticity.h"

#define BOOST_TEST_MODULE SolidMechanicsTest
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE( polynomial_degree_test )
{
    using namespace dealii;
    using namespace Step23;

    deallog.depth_console( 0 );

    double time_step = 2.5e-3;
    double theta = 1;
    unsigned int degree = 1;
    unsigned int n_global_refines = 1;
    double gravity = 2;
    double distributed_load = 0;
    double rho = 1000;
    double final_time = 0.05;

    unsigned int nbComputations = 4;

    std::vector<unsigned int> n_dofs( nbComputations );
    std::vector<double> solution( nbComputations );

    for ( unsigned int i = 0; i < nbComputations; ++i )
    {
        n_global_refines = i + 2;
        LinearElasticity<2> wave_equation_solver( time_step, final_time, theta, degree, gravity, distributed_load, rho, n_global_refines );
        wave_equation_solver.run();

        n_dofs[i] = wave_equation_solver.n_dofs();
        solution[i] = wave_equation_solver.point_value();

        BOOST_CHECK_CLOSE( solution[i], -0.001351993, 0.1 );
    }

    std::vector<double> error( nbComputations - 1 );

    for ( unsigned int i = 0; i < error.size(); ++i )
        error[i] = std::abs( solution[i] - solution[nbComputations - 1] ) / std::abs( solution[nbComputations - 1] );

    for ( unsigned int i = 0; i < error.size() - 1; ++i )
    {
        double rate = 2 * std::log10( error[i] / error[i + 1] );
        rate /= std::log10( n_dofs[i + 1] / n_dofs[i] );

        BOOST_CHECK_GE( rate, 1.9 );
    }
}

BOOST_AUTO_TEST_CASE( polynomial_degree_test_distributed_load )
{
    using namespace dealii;
    using namespace Step23;

    deallog.depth_console( 0 );

    double time_step = 2.5e-3;
    double theta = 1;
    unsigned int degree = 1;
    unsigned int n_global_refines = 1;
    double gravity = 0;
    double distributed_load = 49.757;
    double rho = 1000;
    double final_time = 0.05;

    unsigned int nbComputations = 4;

    std::vector<unsigned int> n_dofs( nbComputations );
    std::vector<double> solution( nbComputations );

    for ( unsigned int i = 0; i < nbComputations; ++i )
    {
        n_global_refines = i + 2;
        LinearElasticity<2> wave_equation_solver( time_step, final_time, theta, degree, gravity, distributed_load, rho, n_global_refines );
        wave_equation_solver.run();

        n_dofs[i] = wave_equation_solver.n_dofs();
        solution[i] = wave_equation_solver.point_value();

        BOOST_CHECK_CLOSE( solution[i], -0.0016910513, 0.1 );
    }

    std::vector<double> error( nbComputations - 1 );

    for ( unsigned int i = 0; i < error.size(); ++i )
        error[i] = std::abs( solution[i] - solution[nbComputations - 1] ) / std::abs( solution[nbComputations - 1] );

    for ( unsigned int i = 0; i < error.size() - 1; ++i )
    {
        double rate = 2 * std::log10( error[i] / error[i + 1] );
        rate /= std::log10( n_dofs[i + 1] / n_dofs[i] );

        BOOST_CHECK_GE( rate, 1.9 );
    }
}

BOOST_AUTO_TEST_CASE( crank_nicolson_distributed_load )
{
    using namespace dealii;
    using namespace Step23;

    deallog.depth_console( 0 );

    double time_step = 2.5e-3;
    double theta = 0.5;
    unsigned int degree = 1;
    unsigned int n_global_refines = 2;
    double gravity = 0;
    double distributed_load = 49.757;
    double rho = 1000;
    double final_time = 0.05;

    unsigned int nbComputations = 4;

    std::vector<unsigned int> nbTimeSteps( nbComputations );
    std::vector<double> solution_l2_norm( nbComputations );

    for ( unsigned int i = 0; i < nbComputations; ++i )
    {
        double dt = time_step / std::pow( 2, i );

        LinearElasticity<2> wave_equation_solver( dt, final_time, theta, degree, gravity, distributed_load, rho, n_global_refines );
        wave_equation_solver.run();

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

    for ( unsigned int i = 0; i < solution_l2_norm.size(); ++i )
        std::cout << "l2norm = " << solution_l2_norm[i] << std::endl;

    for ( unsigned int i = 0; i < error.size(); ++i )
    {
        error[i] = std::abs( solution_l2_norm[i] - solution_l2_norm[nbComputations - 1] ) / std::abs( solution_l2_norm[nbComputations - 1] );

        std::cout << "error = " << error[i] << std::endl;
    }

    std::vector<double> order( nbComputations - 2 );

    for ( unsigned int i = 0; i < order.size(); ++i )
    {
        double dti = time_step / std::pow( 2, i );
        double dtinew = time_step / std::pow( 2, i + 1 );
        order[i] = std::log10( error[i + 1] ) - std::log10( error[i] );
        order[i] /= std::log10( dtinew ) - std::log10( dti );
        std::cout << "order = " << order[i] << std::endl;

        BOOST_CHECK_GE( order[i], 2 );
    }
}

BOOST_AUTO_TEST_CASE( crank_nicolson_combined_load )
{
    using namespace dealii;
    using namespace Step23;

    deallog.depth_console( 0 );

    double time_step = 2.5e-3;
    double theta = 0.5;
    unsigned int degree = 1;
    unsigned int n_global_refines = 2;
    double gravity = 2;
    double distributed_load = 49.757;
    double rho = 1000;
    double final_time = 0.05;

    unsigned int nbComputations = 4;

    std::vector<unsigned int> nbTimeSteps( nbComputations );
    std::vector<double> solution_l2_norm( nbComputations );

    for ( unsigned int i = 0; i < nbComputations; ++i )
    {
        double dt = time_step / std::pow( 2, i );

        LinearElasticity<2> wave_equation_solver( dt, final_time, theta, degree, gravity, distributed_load, rho, n_global_refines );
        wave_equation_solver.run();

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

    for ( unsigned int i = 0; i < solution_l2_norm.size(); ++i )
        std::cout << "l2norm = " << solution_l2_norm[i] << std::endl;

    for ( unsigned int i = 0; i < error.size(); ++i )
    {
        error[i] = std::abs( solution_l2_norm[i] - solution_l2_norm[nbComputations - 1] ) / std::abs( solution_l2_norm[nbComputations - 1] );

        std::cout << "error = " << error[i] << std::endl;
    }

    std::vector<double> order( nbComputations - 2 );

    for ( unsigned int i = 0; i < order.size(); ++i )
    {
        double dti = time_step / std::pow( 2, i );
        double dtinew = time_step / std::pow( 2, i + 1 );
        order[i] = std::log10( error[i + 1] ) - std::log10( error[i] );
        order[i] /= std::log10( dtinew ) - std::log10( dti );
        std::cout << "order = " << order[i] << std::endl;

        BOOST_CHECK_GE( order[i], 2 );
    }
}

BOOST_AUTO_TEST_CASE( crank_nicolson_test )
{
    using namespace dealii;
    using namespace Step23;

    deallog.depth_console( 0 );

    double time_step = 2.5e-3;
    double theta = 0.5;
    unsigned int degree = 1;
    unsigned int n_global_refines = 2;
    double gravity = 2;
    double distributed_load = 0;
    double rho = 1000;
    double final_time = 0.05;

    unsigned int nbComputations = 4;

    std::vector<unsigned int> nbTimeSteps( nbComputations );
    std::vector<double> solution_l2_norm( nbComputations );

    for ( unsigned int i = 0; i < nbComputations; ++i )
    {
        double dt = time_step / std::pow( 2, i );

        LinearElasticity<2> wave_equation_solver( dt, final_time, theta, degree, gravity, distributed_load, rho, n_global_refines );
        wave_equation_solver.run();

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

    for ( unsigned int i = 0; i < solution_l2_norm.size(); ++i )
        std::cout << "l2norm = " << solution_l2_norm[i] << std::endl;

    for ( unsigned int i = 0; i < error.size(); ++i )
    {
        error[i] = std::abs( solution_l2_norm[i] - solution_l2_norm[nbComputations - 1] ) / std::abs( solution_l2_norm[nbComputations - 1] );

        std::cout << "error = " << error[i] << std::endl;
    }

    std::vector<double> order( nbComputations - 2 );

    for ( unsigned int i = 0; i < order.size(); ++i )
    {
        double dti = time_step / std::pow( 2, i );
        double dtinew = time_step / std::pow( 2, i + 1 );
        order[i] = std::log10( error[i + 1] ) - std::log10( error[i] );
        order[i] /= std::log10( dtinew ) - std::log10( dti );
        std::cout << "order = " << order[i] << std::endl;

        BOOST_CHECK_GE( order[i], 2 );
    }
}

BOOST_AUTO_TEST_CASE( backward_euler )
{
    using namespace dealii;
    using namespace Step23;

    deallog.depth_console( 0 );

    double time_step = 2.5e-3;
    double theta = 1;
    unsigned int degree = 1;
    unsigned int n_global_refines = 2;
    double gravity = 2;
    double distributed_load = 0;
    double rho = 1000;
    double final_time = 0.05;

    unsigned int nbComputations = 4;

    std::vector<unsigned int> nbTimeSteps( nbComputations );
    std::vector<double> solution_l2_norm( nbComputations );

    for ( unsigned int i = 0; i < nbComputations; ++i )
    {
        double dt = time_step / std::pow( 2, i );

        LinearElasticity<2> wave_equation_solver( dt, final_time, theta, degree, gravity, distributed_load, rho, n_global_refines );
        wave_equation_solver.run();

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
    {
        error[i] = std::abs( solution_l2_norm[i] - solution_l2_norm[nbComputations - 1] ) / std::abs( solution_l2_norm[nbComputations - 1] );
        std::cout << "error = " << error[i] << std::endl;
    }

    std::vector<double> order( nbComputations - 2 );

    for ( unsigned int i = 0; i < order.size(); ++i )
    {
        double dti = time_step / std::pow( 2, i );
        double dtinew = time_step / std::pow( 2, i + 1 );
        order[i] = std::log10( error[i + 1] ) - std::log10( error[i] );
        order[i] /= std::log10( dtinew ) - std::log10( dti );

        std::cout << "order = " << order[i] << std::endl;

        BOOST_CHECK_GE( order[i], 1 );
    }
}

BOOST_AUTO_TEST_CASE( theta )
{
    using namespace dealii;
    using namespace Step23;

    deallog.depth_console( 0 );

    double time_step = 2.5e-3;
    double theta = 0.6;
    unsigned int degree = 1;
    unsigned int n_global_refines = 2;
    double gravity = 2;
    double distributed_load = 0;
    double rho = 1000;
    double final_time = 0.05;

    unsigned int nbComputations = 4;

    std::vector<unsigned int> nbTimeSteps( nbComputations );
    std::vector<double> solution_l2_norm( nbComputations );

    for ( unsigned int i = 0; i < nbComputations; ++i )
    {
        double dt = time_step / std::pow( 2, i );

        LinearElasticity<2> wave_equation_solver( dt, final_time, theta, degree, gravity, distributed_load, rho, n_global_refines );
        wave_equation_solver.run();

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

        BOOST_CHECK_GE( order[i], 1 );
    }
}

BOOST_AUTO_TEST_CASE( writePositions )
{
    using namespace dealii;
    using namespace Step23;

    double time_step = 2.5e-3;
    double theta = 0.6;
    unsigned int degree = 1;
    unsigned int n_global_refines = 0;
    double gravity = 2;
    double distributed_load = 0;
    double rho = 1000;
    double final_time = 0.05;

    LinearElasticity<2> linear_elasticity_solver( time_step, final_time, theta, degree, gravity, distributed_load, rho, n_global_refines );

    EigenMatrix writePositions;
    linear_elasticity_solver.getWritePositions( writePositions );

    BOOST_CHECK_EQUAL( writePositions.cols(), 2 );
    BOOST_CHECK_GE( writePositions.rows(), 0 );
    BOOST_CHECK_CLOSE( readPositions( 0, 0 ), 0.251109, 0.1 );
    BOOST_CHECK_CLOSE( readPositions( 0, 1 ), 0.19, 0.1 );
    BOOST_CHECK_CLOSE( readPositions( 1, 0 ), 0.2569, 0.1 );
    BOOST_CHECK_CLOSE( readPositions( 1, 1 ), 0.19, 0.1 );
}

BOOST_AUTO_TEST_CASE( readPositions )
{
    using namespace dealii;
    using namespace Step23;

    double time_step = 2.5e-3;
    double theta = 0.6;
    unsigned int degree = 1;
    unsigned int n_global_refines = 0;
    double gravity = 2;
    double distributed_load = 0;
    double rho = 1000;
    double final_time = 0.05;

    LinearElasticity<2> linear_elasticity_solver( time_step, final_time, theta, degree, gravity, distributed_load, rho, n_global_refines );

    EigenMatrix readPositions;
    linear_elasticity_solver.getReadPositions( readPositions );

    BOOST_CHECK_EQUAL( readPositions.cols(), 2 );
    BOOST_CHECK_GE( readPositions.rows(), 0 );
    BOOST_CHECK_CLOSE( readPositions( 0, 0 ), 0.251109, 0.1 );
    BOOST_CHECK_CLOSE( readPositions( 0, 1 ), 0.19, 0.1 );
    BOOST_CHECK_CLOSE( readPositions( 1, 0 ), 0.2569, 0.1 );
    BOOST_CHECK_CLOSE( readPositions( 1, 1 ), 0.19, 0.1 );
}
