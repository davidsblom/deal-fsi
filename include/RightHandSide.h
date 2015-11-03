
#pragma once

/*
 * Author
 *   David Blom, TU Delft. All rights reserved.
 */

namespace Step23
{

    using namespace dealii;

template <int dim>
class RightHandSide :  public Function<dim>
{
public:
  RightHandSide ( double gravity );

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
  private:
      const double gravity;
};

#include "../src/RightHandSide.tpp"

}
