
/*
 * Author
 *   David Blom, TU Delft. All rights reserved.
 */

#pragma once

#include <deal.II/base/parameter_handler.h>

namespace dealiifsi
{
    using namespace dealii;

    class DataStorage : public Subscriptor
    {
public:

        DataStorage();

        ~DataStorage();

        void read_data( const char * );

        int degree, n_global_refines;
        double final_time, time_step, theta;
        double gravity, distributed_load;
        double rho, E, nu;
        bool output_paraview;
        int output_interval;

private:

        void declare_parameters();

        ParameterHandler prm;
    };

    #include "../src/DataStorage.tpp"
}
