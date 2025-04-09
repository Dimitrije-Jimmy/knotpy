import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint as pp
import copy

from knotpy.notation.native import from_knotpy_notation, from_pd_notation
from knotpy.notation.pd import to_pd_notation
from knotpy.drawing.draw_matplotlib import draw, draw_from_layout
from knotpy.utils.geometry import Circle
from knotpy.algorithms.structure import edges

import knotpy as kp
from knotpy.drawing.layout import circlepack_layout

from simulation import SimCircle, extract_circle_positions, extract_main_points_and_connections

from spline import plot_spline_normal
from sequence import process_paths, process_path, plot_splines_sequence#, build_all_circles_data
from over_under_pass import plot_splines, plot_splines_overunder2, build_all_circles_data, build_skip_indices, build_skip_indices_for_all_paths

#__all__ = ['process_paths', 'process_path', 'plot_splines', 'build_all_circles_data']
__version__ = 'god knows'
__author__ = 'Dimitrije Pešić'


# NetworkX implementation __________________________________________________

def build_graph_direct(ret):
    """
    Build a NetworkX graph from the 'ret' dictionary structure.
    - If keys in ret are node labels (strings like 'a', 'b', 'c'),
      we add them as nodes.
    - If keys are frozensets or tuples with 2 items, treat that as an edge.
    """
    G = nx.Graph()
    
    # First add all node keys (like 'a', 'b', 'c')
    # You might also have arcs represented by frozenset(...) or tuples(...).
    # So only add if it is a string (or something that identifies a node).
    for key in ret:
        #print(key)
        if isinstance(key, str):
            G.add_node(key)
    
        # Now parse the frozenset/tuple keys to add edges

        if isinstance(key, (frozenset)):#, tuple)):
            #G.add_node(key)
            nodes = list(key)
            # If exactly 2 items, one edge
            if len(nodes) == 2:
                # Let's try only edge no frozenset
                G.add_edge(str(nodes[0])[0], str(nodes[1])[0])
    
    return G

def build_graph_frozensets(ret):
    """
    Build a NetworkX graph from the 'ret' dictionary structure.
    - If keys in ret are node labels (strings like 'a', 'b', 'c'),
      we add them as nodes.
    - If keys are frozensets or tuples with 2 items, treat that as an edge.
    """
    G = nx.Graph()
    
    # First add all node keys (like 'a', 'b', 'c')
    # You might also have arcs represented by frozenset(...) or tuples(...).
    # So only add if it is a string (or something that identifies a node).
    for key in ret:
        #print(key)
        if isinstance(key, str):
            G.add_node(key)
    
        # Now parse the frozenset/tuple keys to add edges

        if isinstance(key, (frozenset)):
            G.add_node(key)
            nodes = list(key)
            # If exactly 2 items, one edge
            if len(nodes) == 2:
                G.add_edge(str(nodes[0])[0], key)
                G.add_edge(str(nodes[1])[0], key)    
    return G

def build_graph_all(ret):
    """
    Build a NetworkX graph from the 'ret' dictionary structure.
    - takes all points in ret and adds them as nodes
    """
    G = nx.Graph()
    
    # First add all node keys (like 'a', 'b', 'c')
    # You might also have arcs represented by frozenset(...) or tuples(...).
    # So only add if it is a string (or something that identifies a node).
    for key in ret:
        #print(key)
        if isinstance(key, str):
            G.add_node(key)

        elif isinstance(key, frozenset):
            G.add_node(key)
            nodes = list(key)
            if len(nodes) == 2:
                G.add_edge(str(nodes[0])[0], key)
                G.add_edge(str(nodes[1])[0], key)
        
        elif isinstance(key, tuple):
            G.add_node(key)
            nodes = list(key)
            #for i in range(len(nodes) - 1):
            for i in range(len(nodes)):
                G.add_edge(str(nodes[i])[0], key)
    
        else:
            raise ValueError(f"Unknown key type: {type(key)}")

    return G

def extract_positions_forNetworkX(ret, function=build_graph_frozensets):
    new_ret = ret
    if function == build_graph_frozensets:
        new_ret = extract_main_points_and_connections(ret)
    points_from_ret = extract_circle_positions(new_ret) 
    points = {key: (value.real, value.imag) for key, value in points_from_ret.items()}
    
    return points

import networkx as nx

def choose_layout_function(G, small_graph_threshold=20):
    """
    Dynamically choose the layout function based on the graph size.
    For example, if the graph has fewer than `small_graph_threshold` nodes,
    use ARF layout; otherwise, use graphviz with 'neato' or 'sfdp'.
    """
    n = G.number_of_nodes()
    if n < small_graph_threshold:
        #return pos
        return "arf_layout"      # Use ARF for smaller graphs
    else:
        return "graphviz_neato"  # Use Graphviz 'neato' for larger graphs


def plot_networkx_layout(ret, function=build_graph_frozensets, small_graph_threshold=20):
    G = function(ret)
    print(G.edges())

    points = extract_positions_forNetworkX(ret, function)
    print(points)
    
    # Compute a spring layout (force-directed)
    # Vsi attempti kere funkcije delajo kere ne
    #pos = nx.spring_layout(G, seed=42)  # seed for reproducibility
    # Alternatively: pos = nx.kamada_kawai_layout(G)
    #pos = nx.kamada_kawai_layout(G, pos=points) 
    #pos = nx.nx_agraph.graphviz_layout(G, prog='neato') # works great for very large ones
    #pos = nx.nx_agraph.graphviz_layout(G, prog='sfdp')  # works well alternative to neato
    #pos = nx.planar_layout(G)
    #pos = nx.arf_layout(G, pos=points)#, seed=42)      # works best for less
    #pos = nx.spring_layout(G, pos=points)#, seed=42)
    
    """
    Dynamically choose the layout function based on the graph size.
    For example, if the graph has fewer than `small_graph_threshold` nodes,
    use ARF layout; otherwise, use graphviz with 'neato' or 'sfdp'.
    """
    n = G.number_of_nodes()
    if n < small_graph_threshold:
        #return "arf_layout"      # Use ARF for smaller graphs
        pos = nx.arf_layout(G, pos=points)
        #return pos
    else:
        #return "graphviz_neato"  # Use Graphviz 'neato' for larger graphs
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')


    # Plot
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='grey')
    plt.title("NetworkX Force-Directed Layout")
    plt.axis('equal')
    plt.show()

    return pos

import networkx as nx

def layout_positions(G, layout_name="spring", initial_positions=None):
    """
    Return positions (dict: node -> (x, y)) from various layout algorithms.

    Parameters
    ----------
    G : nx.Graph
        The graph object.
    layout_name : str
        Which layout to use ("spring", "kamada_kawai", "graphviz_neato",
        "graphviz_sfdp", "arf_layout", etc.).
    initial_positions : dict, optional
        A dict of node->(x, y) to pass as a starting position if supported
        by the layout function.

    Returns
    -------
    pos : dict
        A dictionary {node: (x, y)} with final positions.
    """
    layout_functions = {
        "spring": lambda g, pos: nx.spring_layout(g, seed=42, pos=pos),
        "kamada_kawai": lambda g, pos: nx.kamada_kawai_layout(g, pos=pos),
        "arf_layout": lambda g, pos: nx.arf_layout(g, pos=pos),
        "graphviz_sfdp": lambda g, pos: nx.nx_agraph.graphviz_layout(g, prog='sfdp'),
        "graphviz_neato": lambda g, pos: nx.nx_agraph.graphviz_layout(g, prog='neato', args='-Goverlap="false"'),
        #"graphviz_neato": lambda g, pos: nx.nx_agraph.graphviz_layout(g, prog='neato', args='-Goverlap="false", -Gdim=3'),
        #"graphviz_neato": lambda g, pos: nx.nx_agraph.graphviz_layout(g, prog='neato'),#, args='-Goverlap="false"'),
        #"graphviz_dot": lambda g, pos: nx.nx_agraph.graphviz_layout(g, prog='dot')#, args='-Goverlap="false"'),    # Just NO
        "planar_layout": lambda g, pos: nx.planar_layout(g),    # Doesn't overlap but makes it weird as fuck
    }

    if layout_name not in layout_functions:
        raise ValueError(f"Unknown layout name: {layout_name}")

    # If the chosen layout doesn't support an initial `pos` argument,
    # we can ignore it or skip passing it. Here, for graphviz, we skip `pos`.
    if layout_name.startswith("graphviz_"):
        return layout_functions[layout_name](G, None)
    else:
        return layout_functions[layout_name](G, initial_positions)

def plot_networkx_layout2(ret, function=build_graph_frozensets, layout_name="arf_layout", small_graph_threshold=20):
    """
    Plot the graph G using the specified layout, and then return the positions.

    Returns
    -------
    pos : dict
        The positions used for plotting (node -> (x, y)).
    """
    G = function(ret)
    #print(G.edges())

    initial_positions = extract_positions_forNetworkX(ret, function)
    #print(initial_positions)
   
    # 1) Obtain positions
    pos = layout_positions(G, layout_name, initial_positions)

    # 2) Plot with matplotlib
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='grey')
    plt.title(f"NetworkX Layout: {layout_name}")
    plt.axis('equal')
    plt.show()

    # 3) Return pos for further usage
    return pos

def prep_networkx_sim_positions(positions_dict, default_radius=0.5):
    """
    Convert a positions dictionary of the form:
        {
           node_or_frozenset: np.array([x, y]),
           ...
        }
    into a dictionary of node_or_frozenset -> SimCircle objects.

    Parameters
    ----------
    positions_dict : dict
        Keys can be strings or frozensets or tuples,
        Values are 2D coordinates (e.g. np.array([x, y])).
    default_radius : float
        The radius used for each created SimCircle.

    Returns
    -------
    circlepack_layout_sim : dict
        Dictionary where each key is the original node/frozenset,
        and each value is a SimCircle object with center and radius.
    """
    circlepack_layout_sim = {}

    for key, coord_array in positions_dict.items():
        # Ensure we have something like [x, y]
        # Convert to complex for the Circle center
        #x, y = coord_array[0], coord_array[1]
        x, y = (coord_array)
        center = complex(x, y)

        sim_circle = SimCircle(center, default_radius)
        circlepack_layout_sim[key] = sim_circle

    return circlepack_layout_sim

import numpy as np

def prep_sim_positions(positions_dict, default_radius=0.5):
    """
    Convert a positions dictionary of the form:
      {
        node_or_frozenset: (x, y)  or  np.array([x, y]),
        ...
      }
    into a dictionary of node_or_frozenset -> SimCircle.
    """
    circlepack_layout_sim2 = {}

    for key, pos in positions_dict.items():
        # 1) Ensure a consistent array-like format
        if isinstance(pos, tuple):
            pos = np.array(pos, dtype=float)

        print(pos)
        # 2) Convert to complex
        x, y = pos
        print(x, y)
        center = complex(x, y)

        sim_circle = SimCircle(center, default_radius)
        circlepack_layout_sim2[key] = sim_circle

    return circlepack_layout_sim2

import numpy as np

def recenter_positions(positions_dict):
    """
    Take a dictionary { key: (x, y) } or { key: np.array([x, y]) }
    and shift all coordinates so the center is at (0,0).

    Returns a new dictionary { key: np.array([x, y]) }.
    """
    # Convert all positions to NumPy arrays for easy manipulation
    positions_array = {}
    for k, pos in positions_dict.items():
        positions_array[k] = np.array(pos, dtype=float)

    # Compute the centroid (mean of all x's and y's)
    xs = [pos[0] for pos in positions_array.values()]
    ys = [pos[1] for pos in positions_array.values()]
    center_x = np.mean(xs)
    center_y = np.mean(ys)

    # Shift positions so centroid is at (0,0)
    recentered = {}
    for k, pos in positions_array.items():
        new_x = pos[0] - center_x
        new_y = pos[1] - center_y
        recentered[k] = np.array([new_x, new_y], dtype=float)

    return recentered

def prep_sim_positions(positions_dict, default_radius=0.5):
    """
    Convert positions (x,y) or np.array([x,y]) into SimCircle objects,
    after re-centering them around (0,0) if desired.
    """
    # Optionally recenter the positions:
    positions_dict = recenter_positions(positions_dict)

    circlepack_layout_sim2 = {}

    for key, pos in positions_dict.items():
        # Ensure a consistent array
        pos = np.array(pos, dtype=float)

        x, y = pos
        center = complex(x, y)  # or keep as array if you prefer

        sim_circle = SimCircle(center, default_radius)
        circlepack_layout_sim2[key] = sim_circle

    return circlepack_layout_sim2


# Usage example:

if __name__ == "__main__":
    # Define your knot using PD notation
    s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
    # Large:
    s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
    #s = "V[0,1,2], V[0,2,1]"
    #s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
    ## Problematic:
    s = "X[0,1,2,3],X[4,5,6,7],X[1,8,9,10],X[11,12,13,9],X[14,15,7,16],X[17,18,19,13],X[10,20,17,12],X[20,19,15,14],V[3,21,22],V[5,23,24],V[6,24,16],V[4,18,23],V[0,22,8],V[2,11,21]"
    k2 = from_pd_notation(s)
    print(k2)

    ret = circlepack_layout(k2)



    #plot_networkx_layout(ret, function=build_graph_all)         # nah just not useful
    #plot_networkx_layout(ret, function=build_graph_frozensets)  # best
    #plot_networkx_layout(ret, function=build_graph_direct)     # nah just not useful

    #positions = plot_networkx_layout2(ret, function=build_graph_frozensets, layout_name="arf_layout")
    #positions = plot_networkx_layout2(ret, function=build_graph_frozensets, layout_name="graphviz_sfdp")
    #positions = plot_networkx_layout2(ret, function=build_graph_frozensets, layout_name="graphviz_neato")
    #positions = plot_networkx_layout2(ret, function=build_graph_frozensets, layout_name="planar_layout")
    
    positions = plot_networkx_layout(ret, function=build_graph_frozensets)#, layout_name="kamada_kawai")
    pp.pprint(positions)

    positions_sim = prep_networkx_sim_positions(positions)
    positions_sim2 = prep_sim_positions(positions)
    pp.pprint(positions_sim2)

    # Plot initial spline curves
    print("\nPaths:")
    paths = edges(k2)
    print(paths)
    processed_paths = process_paths(paths, ret)
    print(processed_paths)
    circles_data = build_all_circles_data(processed_paths, positions_sim2)
    #circles_data2 = build_all_circles_data(processed_paths, ret2)
    #circles_data3 = build_all_circles_data(processed_paths, ret3)
    print("\nSplines Circles data")
    #pp.pprint(circles_data0)
    #skip_indices = build_skip_indices(sequences=processed_paths, skip_radius=1)    # deprecated
    skip_indices_for_all_paths = build_skip_indices_for_all_paths(processed_paths)

    #plot_splines(circles_data, skip_indices_for_all_paths, gap_radius=10)
    plot_splines_overunder2(circles_data, skip_indices_for_all_paths, gap_fraction=0.02 )#, gap_radius=10)
    #plot_splines_sequence(circles_data)

    plt.show()