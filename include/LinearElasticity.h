
#pragma once

/*
 * Author
 *   David Blom, TU Delft. All rights reserved.
 */

namespace Step23
{
    using namespace dealii;

    template <class Scalar>
    Vector<Scalar> operator *(
        SparseMatrix<Scalar> & A,
        Vector<Scalar> & b
        );


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
    class LinearElasticity
    {
public:

        WaveEquation(
            double time_step,
            double theta,
            double degree,
            double gravity,
            double distributed_load,
            unsigned int n_global_refines
            );
        void run();

        void assemble_system();

        void finalizeTimeStep();

        void initTimeStep();

        void isRunning();

        void setup_system();

        void solve();

        void solve_u();

        void solve_v();

        void output_results() const;

        unsigned int n_dofs() const;

        double point_value() const;

        double get_traction(
            const unsigned int component_i,
            const unsigned int
            );

        unsigned int deg;
        unsigned int n_global_refines;

        Triangulation<dim>   triangulation;
        FESystem<dim>        fe;
        DoFHandler<dim>      dof_handler;

        ConstraintMatrix constraints;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> mass_matrix;
        SparseMatrix<double> laplace_matrix;
        SparseMatrix<double> matrix_u;
        SparseMatrix<double> matrix_v;

        Vector<double>       solution_u, solution_v;
        Vector<double>       old_solution_u, old_solution_v;
        Vector<double>       system_rhs;
        Vector<double>       body_force;
        Vector<double>       old_body_force;

        double time, time_step;
        unsigned int timestep_number;
        const double theta;
        const double gravity, distributed_load;

private:

        bool init;
    };

    #include "../src/LinearElasticity.tpp"
}
