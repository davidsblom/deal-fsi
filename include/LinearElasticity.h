
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

    template <int dim>
    class LinearElasticity
    {
public:

        LinearElasticity(
            double time_step,
            double final_time,
            double theta,
            double degree,
            double gravity,
            double distributed_load,
            double rho,
            unsigned int n_global_refines
            );

        void assemble_system();

        void finalizeTimeStep();

        double get_traction(
            const unsigned int component_i,
            const unsigned int
            );

        void initTimeStep();

        bool isRunning();

        void output_results() const;

        unsigned int n_dofs() const;

        double point_value() const;

        void run();

        void setup_system();

        void solve();

        void solve_u();

        void solve_v();

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
        double initial_time, final_time, rho;
    };

    #include "../src/LinearElasticity.tpp"
}
