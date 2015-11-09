
/*
 * Author
 *   David Blom, TU Delft. All rights reserved.
 */

template <int dim>
RightHandSide<dim>::RightHandSide ( double gravity )
    :
    Function<dim> ( dim ),
    gravity( gravity )
{}

template <int dim>
inline
void RightHandSide<dim>::vector_value(
    const Point<dim> &,
    Vector<double> & values
    ) const
{
    Assert( values.size() == dim,
        ExcDimensionMismatch( values.size(), dim ) );
    Assert( dim >= 2, ExcNotImplemented() );

    double rho = 1000;
    values( 0 ) = 0;
    values( 1 ) = -gravity * rho;

    double t = this->get_time();
    double T = 0.01;
    double offset = 0.01;

    if ( t - offset < T )
        values( 1 ) *= 0.5 - 0.5 * std::cos( M_PI * (t - offset) / T );

    if ( t < offset )
        values( 1 ) = 0.0;
}

template <int dim>
void RightHandSide<dim>::vector_value_list(
    const std::vector<Point<dim> > & points,
    std::vector<Vector<double> > & value_list
    ) const
{
    Assert( value_list.size() == points.size(),
        ExcDimensionMismatch( value_list.size(), points.size() ) );

    const unsigned int n_points = points.size();

    for ( unsigned int p = 0; p < n_points; ++p )
        RightHandSide<dim>::vector_value( points[p],
            value_list[p] );
}
