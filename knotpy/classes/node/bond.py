from knotpy.classes.node.node import Node

__all__ = ['Bond']
__version__ = '0.1'
__author__ = 'Boštjan Gabrovšek <bostjan.gabrovsek@gmail.si>'


class Bond(Node):
    """a bond is considered to be a small arc (in form of a node) connecting two other arcs.
    The endpoints 0 and 1 are connected. and the endpoints 2 and 3 are connected."""
    def __init__(self, incoming_node_data=None, degree=4, **attr):
        degree = degree or 4
        if degree != 4:
            raise ValueError("Cannot create a bond with degree not equal to four.")

        super().__init__(incoming_node_data, degree=degree, **attr)

    # def mirror(self):
    #     pass

    # @staticmethod
    # def is_crossing(self):
    #     return False

    # @staticmethod
    # def is_bivalent(self):
    #     return False

    def __str__(self):
        return "B" + super().__str__()


# class OrientedBond(Bond):
#     def __init__(self, incoming_node_data=None, degree=4, parallel=True, **attr):
#         degree = degree or 4
#         if degree != 4:
#             raise ValueError("Cannot create a bond with degree not equal to four.")
#
#         super().__init__(incoming_node_data, degree=degree, **attr)
#         self.parallel = parallel
#
#
#     def __str__(self):
#         return "ON" + super().__str__()