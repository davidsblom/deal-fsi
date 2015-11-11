
#pragma once

/*
 * Author
 *   David Blom, TU Delft. All rights reserved.
 */

namespace dealiifsi
{
    using namespace dealii;

    template <int dim>
    class RightHandSide : public Function<dim>
    {
public:

        RightHandSide ( double gravity );

        virtual void vector_value(
            const Point<dim> & p,
            Vector<double> & values
            ) const;

        virtual void vector_value_list(
            const std::vector<Point<dim> > & points,
            std::vector<Vector<double> > & value_list
            ) const;

private:

        const double gravity;
    };

    #include "../src/RightHandSide.tpp"
}
