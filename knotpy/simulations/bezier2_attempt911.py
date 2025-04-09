import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly

# Dictionary of points with keys as 'a', 'b', etc., and values as complex numbers representing x and y
points_dict = {
    'a': complex(0, 0),
    'b': complex(1, 2),
    'c': complex(3, 3),
    'd': complex(4, 0),
    'e': complex(2, -2)
}

# Imports and funcs for PD ________________________________________________________

from knotpy.drawing.circlepack import CirclePack
from knotpy.drawing.draw_matplotlib import draw_from_layout
from knotpy.notation.native import from_knotpy_notation
from knotpy.notation.pd import from_pd_notation
from knotpy.drawing.layout import circlepack_layout
print("pre import")
from spline7_rabmsim import run_sim as simulation
print("post import")
def extract_circle_positions(circle_dict):
    """Extract circle centers from the dictionary of circles."""
    positions = {}
    for key, circle in circle_dict.items():
        positions[key] = circle.center  # Each circle has a .center attribute
    return positions

def extract_main_points_and_connections(ret):
    return {k: v for k, v in ret.items() if isinstance(k, (str, frozenset))}

# End of imports ________________________________________________________________

def extract_points(points_dict, keys):
    """Extracts x and y coordinates from the points dictionary based on the provided keys."""
    points = [points_dict[key] for key in keys]
    x = [p.real for p in points]
    y = [p.imag for p in points]
    return x, y

def bernstein_poly(n, k, t):
    """Calculates the Bernstein polynomial of n, k as a function of t."""
    from scipy.special import comb
    return comb(n, k) * (t ** k) * ((1 - t) ** (n - k))

def bezier_curve_custom(points_dict, keys, num=200):
    """Generates a Bézier curve using custom implementation."""
    x_points, y_points = extract_points(points_dict, keys)
    n = len(x_points) - 1
    t = np.linspace(0, 1, num)
    curve_x = np.zeros_like(t)
    curve_y = np.zeros_like(t)
    for k in range(n + 1):
        bern_poly = bernstein_poly(n, k, t)
        curve_x += x_points[k] * bern_poly
        curve_y += y_points[k] * bern_poly
    return curve_x, curve_y

def bezier_curve_scipy(points_dict, keys, num=200):
    """Generates a Bézier curve using scipy's BPoly."""
    x_points, y_points = extract_points(points_dict, keys)
    t = np.linspace(0, 1, num)
    # Coefficients need to be of shape (number of intervals, degree + 1)
    # Since we have one interval from 0 to 1, the number of intervals k = 1
    # So coefficients should be of shape (1, degree + 1)
    c_x = np.array([x_points])  # Shape (1, n+1)
    c_y = np.array([y_points])  # Shape (1, n+1)
    x = np.array([0, 1])  # Breakpoints
    # Create BPoly
    bezier_x = BPoly(c_x, x)(t)
    bezier_y = BPoly(c_y, x)(t)
    return bezier_x, bezier_y

def plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Bézier Curve", loop=False):
    """Plots the Bézier curve along with control points."""
    plt.figure(figsize=(8, 6))
    # Plot control points
    plt.plot(x_points, y_points, 'ro--', label='Control Points')
    # Plot Bézier curve
    plt.plot(curve_x, curve_y, 'b-', label='Bézier Curve', linewidth=2)
    # If loop, handle overlapping lines
    if loop:
        # Split the curve into segments where the top line can pass over the bottom
        mid_index = len(curve_x) // 2
        plt.plot(curve_x[:mid_index], curve_y[:mid_index], 'b-', linewidth=2)
        plt.plot(curve_x[mid_index:], curve_y[mid_index:], 'b-', linewidth=2)
        plt.plot(curve_x[mid_index - 10:mid_index + 10], curve_y[mid_index - 10:mid_index + 10], 'w', linewidth=4)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Example usage:

# Linear Bézier Curve (2 points)
keys_linear = ['a', 'b']
x_points, y_points = extract_points(points_dict, keys_linear)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_linear)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Linear Bézier Curve")

# Quadratic Bézier Curve (3 points)
keys_quadratic = ['a', 'b', 'c']
x_points, y_points = extract_points(points_dict, keys_quadratic)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_quadratic)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Quadratic Bézier Curve")

# Cubic Bézier Curve (4 points)
keys_cubic = ['a', 'b', 'c', 'd']
x_points, y_points = extract_points(points_dict, keys_cubic)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_cubic)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Cubic Bézier Curve")

# Quartic Bézier Curve (5 points)
keys_quartic = ['a', 'b', 'c', 'd', 'e']
x_points, y_points = extract_points(points_dict, keys_quartic)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_quartic)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Quartic Bézier Curve")

# Bézier Curve with Loop
keys_loop = ['a', 'b', 'e', 'd', 'c']
x_points, y_points = extract_points(points_dict, keys_loop)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_loop)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Bézier Curve with Loop", loop=True)

# Using scipy's BPoly
x_points, y_points = extract_points(points_dict, keys_cubic)
#curve_x_scipy, curve_y_scipy = bezier_curve_scipy(points_dict, keys_cubic)
#plot_bezier_curve(curve_x_scipy, curve_y_scipy, x_points, y_points, title="Cubic Bézier Curve (scipy)")

# Quintic Bézier Curve (6 points)
points_dict['f'] = complex(5, 2)
keys_quintic = ['a', 'b', 'c', 'd', 'e', 'f']
x_points, y_points = extract_points(points_dict, keys_quintic)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_quintic)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Quintic Bézier Curve")

#plt.show()

# Using matplotlib.path and matplotlib.patches ______________________________________________

import matplotlib.patches as patches
from matplotlib.path import Path

def bezier_curve_matplotlib(points_dict, keys):
    """Generates a Bézier curve using matplotlib's Path."""
    points = [points_dict[key] for key in keys]
    verts = [(p.real, p.imag) for p in points]
    if len(verts) == 4:
        codes = [Path.MOVETO,
                 Path.CURVE4,
                 Path.CURVE4,
                 Path.CURVE4]
    elif len(verts) == 3:
        codes = [Path.MOVETO,
                 Path.CURVE3,
                 Path.CURVE3]
    else:
        raise ValueError("Matplotlib's Path supports quadratic (3 points) and cubic (4 points) Bézier curves.")
    path = Path(verts, codes)
    return path

def plot_bezier_curve_matplotlib(path, x_points, y_points, title="Bézier Curve"):
    """Plots the Bézier curve using matplotlib's PathPatch."""
    fig, ax = plt.subplots()
    patch = patches.PathPatch(path, facecolor='none', lw=2, edgecolor='blue')
    ax.add_patch(patch)
    ax.plot(x_points, y_points, 'ro--', label='Control Points')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.show()

# Example usage:
keys_cubic = ['a', 'b',  'd']
x_points, y_points = extract_points(points_dict, keys_cubic)
path = bezier_curve_matplotlib(points_dict, keys_cubic)
#plot_bezier_curve_matplotlib(path, x_points, y_points, title="Cubic Bézier Curve (matplotlib)")

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import pprint

pp = pprint.PrettyPrinter(indent=4)

def build_graph(ret):
    graph = defaultdict(list)
    circles = {}
    identifiers = set()

    for key, circle in ret.items():
        if isinstance(key, frozenset):
            id_from, id_to = key
            node_from = str(id_from)[0]
            node_to = str(id_to)[0]
            # Store both directions in the graph
            graph[node_from].append((node_to, key))  # Edge from node_from to node_to
            graph[node_to].append((node_from, key))  # Edge from node_to to node_from
            # Map the connection to its circle
            circles[key] = circle
            identifiers.update([id_from, id_to])
        elif isinstance(key, str):
            # Key is a node
            node = str(key)
            circles[node] = circle
            identifiers.add(node)

    pp.pprint(graph)

    return graph, circles, identifiers

def extract_unique_connections(graph):
    unique_connections = set()
    connections_list = []

    for node_from, edges in graph.items():
        for node_to, control_points in edges:
            # Create a sorted tuple to avoid duplicates
            connection = tuple(sorted([node_from, node_to]) + [control_points])
            if connection not in unique_connections:
                unique_connections.add(connection)
                connections_list.append([node_from, control_points, node_to])
    return connections_list

def get_positions(circles, elements):
    """Retrieve positions from circles dictionary for the given elements."""
    positions = []
    for elem in elements:
        positions.append(circles[elem])
    return positions

def bezier_curve_matplotlib_multiple(connections, circles):
    """Plots multiple Bezier curves on the same plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for connection in connections:
        start_node, control_points_set, end_node = connection
        # Get positions
        start_pos = circles[start_node]
        end_pos = circles[end_node]
        control_points = list(control_points_set)
        # Sort control points if necessary
        control_points.sort()  # Implement a better sorting if needed
        #control_positions = [circles[cp] for cp in control_points]
        control_positions = circles[control_points_set]     # lets hope
        
        # Build vertices and codes for the Bezier curve
        verts = [(start_pos.real, start_pos.imag)]
        codes = [Path.MOVETO]
        """
        if len(control_positions) == 1:
            # Quadratic Bezier Curve
            verts.extend([(control_positions[0].real, control_positions[0].imag),
                          (end_pos.real, end_pos.imag)])
            codes.extend([Path.CURVE3, Path.CURVE3])
        elif len(control_positions) == 2:
            # Cubic Bezier Curve
            verts.extend([(control_positions[0].real, control_positions[0].imag),
                          (control_positions[1].real, control_positions[1].imag),
                          (end_pos.real, end_pos.imag)])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
        else:
            # Handle higher-order Bezier curves if necessary
            print(f"Unsupported number of control points: {len(control_positions)}")
            continue  # Skip this curve
        """
        # Quadratic Bezier Curve
        verts.extend([(control_positions.real, control_positions.imag),
                        (end_pos.real, end_pos.imag)])
        codes.extend([Path.CURVE3, Path.CURVE3])

        
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2, edgecolor='blue')
        ax.add_patch(patch)
        # Plot control points and lines
        all_points = [start_pos] + [control_positions] + [end_pos]
        x_points = [p.real for p in all_points]
        y_points = [p.imag for p in all_points]
        #ax.plot(x_points, y_points, 'ro--', markersize=4)

    ax.set_title("Bezier Curves for Graph Connections")
    ax.grid(True)
    ax.axis('equal')
    plt.show()

# Sample data
ret = {
    'a': complex(0, 0),
    'b': complex(4, 0),
    'c': complex(2, 3),
    frozenset({'a1', 'b0'}): complex(2, 0),  # Control point between 'a' and 'b'
    frozenset({'a0', 'c1'}): complex(1, 1.5),  # Control point between 'a' and 'c'
    frozenset({'c2', 'a3'}): complex(3, 1.5),  # Another control point between 'a' and 'c'
    frozenset({'b1', 'c0'}): complex(3, 1.5),  # Control point between 'b' and 'c'
    frozenset({'b2', 'c3'}): complex(1, 1.5),  # Another control point between 'b' and 'c'
    frozenset({'b3', 'a2'}): complex(2, -1),   # Control point between 'a' and 'b' (second one)
    # Add more if needed
}

s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
#s = "V[0,1,2], V[0,2,1]"
#s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
# problematic
s = "X[0,1,2,3],X[4,5,6,7],X[1,8,9,10],X[11,12,13,9],X[14,15,7,16],X[17,18,19,13],X[10,20,17,12],X[20,19,15,14],V[3,21,22],V[5,23,24],V[6,24,16],V[4,18,23],V[0,22,8],V[2,11,21]"
k2 = from_pd_notation(s)
print(k2)

ret = circlepack_layout(k2)

print("\n")

filtered_ret = extract_main_points_and_connections(ret)
ret1 = simulation(filtered_ret)
ret1 = {key: ret[key].center for key in ret}
pp.pprint(ret1)

# Build graph
graph, circles, identifiers = build_graph(ret1)

# Extract unique connections
connections = extract_unique_connections(graph)

# Plot Bezier curves
draw_from_layout(k2, ret, with_labels=True, with_title=True)
bezier_curve_matplotlib_multiple(connections, circles)


"""
TYPING HERE cause I can't be bothered to open up my paper notebook on train

This way it's not going to work, plotting the knots myself is never going
 to turn out well, it'll always be inferior to the pre written thing he has
 (in this small timeframe) thus it won't be achieving the goal of improvement
I will be submitting this as testament of work so he sees that something was
 attempted but failed unfortunately

Next line of reasoning, first of all write this with the simulation integrated
 spline6_411Eulerian2.py skup s ttim sam da zgleja mal vec narjen
 clean up both that and this code of course

Next line of pursuit, going back to the idea of increasing/ decreasing radiuses
 and making sure the node circes are touching and connecting.
 I can maybe reuse some of the functions from here for improvement to the sim

Hopefully now I have a fixed keyboard and don't have to break my head for
 it to work ffs
Negative 

"""