from knotpy.classes.planardiagram import PlanarDiagram

# from knotpy.classes.endpoint import Endpoint

#__all__ = ['choose_kink', 'choose_poke', "check_faces_sanity", "pokes"]
__version__ = '0.1'
__author__ = 'Boštjan Gabrovšek <bostjan.gabrovsek@fs.uni-lj.si>'

#
# def faces(k: PlanarDiagram, length=None):
#     """Iterates over faces (planar graph region/areas) of a planar diagram. A planar region is defined as a sequence of
#     endpoints in the following manner: an endpoint (a, p) in a region R is the endpoint before the crossing a from R in
#     ccw order, in other words, if we turn at a crossing ccw, we get the endpoint (a, p - 1), which forms an arc with
#     (b, q), where (b, q) is again in R.
#     The function takes the set of all endpoints, select an unused endpoint and travels along the edges, until there are no endpoints left.
#     :param k: knotted object
#     :param length: of the length is given, only regions of this length (order) will be considered
#     """
#
#     #print("Computing regions of", k)
#
#     unused_endpoints = set(k.endpoints)
#     while unused_endpoints:
#         ep = unused_endpoints.pop()
#         region = list()
#         while True:
#             region.append(ep)
#             ep = k.nodes[ep.node][(ep.position - 1) % len(k.nodes[ep.node])]
#             if ep in unused_endpoints:
#                 unused_endpoints.remove(ep)
#             else:
#                 break
#         if not length or len(region) == length:
#             yield tuple(region)


def check_faces_sanity(k: PlanarDiagram):
    """
    Check if faces do not overlap (e.g. ccw vertex order respected)
    :param k:
    :return:
    TODO: fix so it works for handcuff-like links (with a cut-edge)
    """
    def unique(s):
        return len(set(s)) == len(s)
    #print(k.faces)
    # for f in k.faces:
    #     print(f)
    # print(list(k.faces))
    regs = list(k.faces)
    return all(unique([ep.node for ep in r]) for r in regs)




# def choose_kink(k):
#     """Returns the first kink region of the knotted planar diagram object k.
#     :return: the singleton kink region if a kink exists, None otherwise
#     """
#     try:
#         return next(iter(kinks(k)))
#     except StopIteration:
#         return None






    # visited_nodes = set()
    # for node in k.crossings:
    #     visited_nodes.add(node)
    #     for ep in k.crossings[node]:
    #         adj_ep = k.nodes[ep.node][(ep.position + 3) & 3]   # the cw endpoint
    #         # the adjacent crossing must not be already visited and must not be the same (not a kink),
    #         # the cw rotation of the ccw adjacent endpoint must be the original endpoint and parities do not match
    #         if ep.node not in visited_nodes and adj_ep.node == node != ep.node and \
    #                 k.nodes[ep.node].is_crossing() and (adj_ep.position + ep.position) & 1:
    #             yield [ep, adj_ep]
