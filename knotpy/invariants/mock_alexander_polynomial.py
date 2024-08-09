from collections import deque
from sympy import Expr, expand, Integer, symbols, Symbol
from itertools import product

from knotpy.classes.planardiagram import PlanarDiagram
from knotpy.algorithms.orientation import oriented
from knotpy.algorithms.skein import smoothen_crossing
from knotpy.algorithms.node_operations import name_for_new_node
from knotpy.classes.node import Crossing, Vertex, Terminal
from knotpy.classes.endpoint import Endpoint, OutgoingEndpoint, IngoingEndpoint
from knotpy.algorithms.components_disjoint import add_unknot
from knotpy.invariants.writhe import writhe
from knotpy.reidemeister.reidemeister import make_all_reidemeister_moves
from knotpy.reidemeister.reidemeister_1 import reidemeister_1_add_kink, reidemeister_1_remove_kink


def mock_alexander_polynomial(k:PlanarDiagram, variable="W"):
    W = variable if isinstance(variable, Symbol) else symbols(variable)

    k = k if k.is_oriented() else oriented(k)

    # get terminal outgoing endpoint
    out_ep = [ep for ep in k.endpoints if k.degree(ep.node) == 1 and type(ep) is OutgoingEndpoint][0]
    # out_ep = None
    # for node in k.vertices:
    #     if k.degree(node) == 1 and type(ep := k.endpoint_from_pair((node, 0))) == OutgoingEndpoint:
    #         out_ep = ep
    #         break
    # if out_ep is None:
    #     raise ValueError("Cannot find outoing endpoint")

    unstarred_faces = [face for face in k.faces if out_ep not in face]
    stack = deque([(Integer(1), set())])
    for face in unstarred_faces:
        new_stack = deque()
        while stack:
            #face_marked = False
            weight, marked_vertices = stack.pop()
            for ep in face:
                if ep.node not in marked_vertices and type(k.nodes[ep.node]) is Crossing:

                    # consider ep as new mark
                    b_outgoing = type(ep) is OutgoingEndpoint  # is the endpoint outgoing?
                    b_over = ep.position % 2  # is the endpoint an overstrand?
                    b_positive = k.nodes[ep.node].sign() > 0  # is the sign positive?

                    if b_outgoing and not b_over and b_positive:
                        new_weight = W
                    elif not b_outgoing and b_over and not b_positive:
                        new_weight = -W
                    elif b_outgoing and b_over and not b_positive:
                        new_weight = W**(-1)
                    elif not b_outgoing and not b_over and b_positive:
                        new_weight = -W**(-1)
                    else:
                        new_weight = Integer(1)

                    new_stack.append((weight * new_weight, marked_vertices | {ep.node, }))

        stack = new_stack

    # print("--")
    # for w, m in stack:
    #     print(w, m)
    # print("--")

    polynomial = sum(w for w,_ in stack)
    return expand(polynomial)





    pass

if __name__ == '__main__':
    import knotpy as kp

    k = kp.from_pd_notation("X[0,4,1,5],X[5,1,6,2],X[2,6,3,7],X[8,4,7,3],V[0],V[8]")
    print(k)
    print(mock_alexander_polynomial(k))

    #k = kp.from_pd_notation("X[3,2,4,1],X[2,5,3,4],V[1],V[5]")
    print(k)
    print(mock_alexander_polynomial(k))
    print("---")
    # ooo = kp.all_orientations(k)
    # o = ooo[0]

    # TODO: make oriented Reidemeister moves

    for o in kp.all_orientations(k):
        print(o)
        print(mock_alexander_polynomial(o))


    for k_r in make_all_reidemeister_moves(k, [reidemeister_1_add_kink], depth=2):
        #print(k_r)
        for o in kp.all_orientations(k_r):
            # print(o)
            # print(kp.to_pd_notation(o))
            print(mock_alexander_polynomial(o))