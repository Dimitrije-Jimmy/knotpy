"""Generators for some simple planar diagrams."""

#__all__ = [""]


from knotpy.notation.em import to_em_notation
from knotpy.classes.planargraph import PlanarGraph


__all__ = ['empty_pd', 'trivial_theta_curve', 'parallel_edge', 'vertex_graph', 'house_graph', 'empty_graph']
__version__ = '0.1'
__author__ = 'Boštjan Gabrovšek'


def empty_pd(n=0, create_using=None):
    """Returns the empty planar diagram with n nodes and zero arcs."""

    if create_using is None:
        pd = PlanarGraph()
    elif type(create_using) is type:
        pd = create_using()
    elif not hasattr(create_using, "adj"):
        raise TypeError("create_using is not data valid PlanarDiagram  type or instance")
    else:
        # create_using is data PD style SpacialGraph
        create_using.clear()
        pd = create_using

    pd.add_nodes(n)
    return pd


def trivial_theta_curve():
    pg = PlanarGraph()
    pg.add_node(0, (0, 2, 1))
    pg.add_node(1, (0, 1, 2))
    return pg


def parallel_edge(multiplicity):
    """Returns data graph with two vertices and multiplicity edges between the two nodes."""
    pg = PlanarGraph()
    pg.add_node(0, multiplicity)
    pg.add_node(1, multiplicity)
    if multiplicity > 0:
        for i in range(multiplicity):
            pg.add_arc((0, i), (1, multiplicity-i-1))
        return pg

def house_graph():
    """
    4--3-\
    |  |  2
    0--1-/
    """

    g = PlanarGraph()
    g.add_node(0,degree=2)
    g.add_node(1,degree=3)
    g.add_node(2,degree=2)
    g.add_node(3,degree=3)
    g.add_node(4,degree=2)
    g.add_arc((0,0), (1,0))
    g.add_arc((1,1), (2,0))
    g.add_arc((2,1), (3,1))
    g.add_arc((3,2), (4,0))
    g.add_arc((4,1), (0,1))
    g.add_arc((1,2), (3,0))


    return g



def empty_graph():
    return PlanarGraph()


def vertex_graph():
    pg = PlanarGraph()
    pg.add_node(0, degree=0)
    return pg

if __name__ == '__main__':
    print(empty_graph())
    print(vertex_graph())
    print(to_em_notation(empty_graph()))
    print(to_em_notation(vertex_graph()))


"""
for m in range(4, 6):
    g = parallel_edge(m)
    print(g)
    em = to_em_notation(g)
    print(from_notation(em))
    data = to_pd_notation(g)
    print(from_notation(data))
"""