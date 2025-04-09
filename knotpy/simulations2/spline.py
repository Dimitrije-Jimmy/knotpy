"""
Spline functions for plotting
"""
import numpy as np
import matplotlib.pyplot as plt
#import scipy.interpolate as scipy.interpolate
from tqdm import tqdm
import pprint as pp

from knotpy.notation.native import from_knotpy_notation, from_pd_notation
from knotpy.notation.pd import to_pd_notation
from knotpy.drawing.draw_matplotlib import draw, draw_from_layout
from knotpy.utils.geometry import Circle

import knotpy as kp
from knotpy.drawing.layout import circlepack_layout

__all__ = ['plot_spline_normal', 'build_graph', 'build_circles_data', 'traverse_knot', 'extract_points', 
           'plot_bezier_curve_matplotlib', 'bezier_curve_matplotlib', 'bezier_curve_matplotlib_multiple', 'get_positions']
__version__ = 'god knows'



# Using matplotlib.path and matplotlib.patches ______________________________________________
def extract_points(points_dict, keys):
    """Extracts x and y coordinates from the points dictionary based on the provided keys."""
    points = [points_dict[key] for key in keys]
    x = [p.real for p in points]
    y = [p.imag for p in points]
    return x, y

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

    print(f"\nGraph")
    pp.pprint(graph)

    return graph, circles, identifiers

def extract_unique_connections(graph):
    """
    Extracts unique connections from a graph represented as an adjacency list.
    
    Parameters:
        graph (dict): Adjacency list, where each key is a node and its value is a list of tuples, each containing the neighbor node and a set of control points.
    
    Returns:
        list: List of unique connections, where each connection is a list of three elements: the node the connection starts at, the set of control points, and the node the connection ends at.
    """
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
    fig, ax = plt.subplots(figsize=(6, 6))
    
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

# End of Bezier functions _____________________________________________________

# Start of Sequencing Spline __________________________________________________

def traverse_knot(graph, start_node):
    visited_edges = set()
    sequence = []

    def dfs(node):
        sequence.append(node)
        for neighbor, conn_key in graph[node]:
            edge = frozenset({node, neighbor, conn_key})
            if edge not in visited_edges:
                visited_edges.add(edge)
                sequence.append(conn_key)  # Add the connection
                dfs(neighbor)
                # Optional: sequence.append(node)  # If you want to record returning to the node

    dfs(start_node)
    print(sequence)
    return sequence



def build_circles_data(sequence, circles):
    circles_data = []
    for key in sequence:
        circle = circles.get(key)
        #circle = circles[key]
        if circle:
            center = circle.center
            radius = circle.radius
            circles_data.append((center, radius))
        else:
            print(f"Warning: Circle not found for key {key}")
    return circles_data

def plot_spline_normal(circles_data, num_spline_points=300):
    from scipy.interpolate import make_interp_spline
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract centers from circles_data
    centers = [center for center, _ in circles_data]
    x_points = np.array([center.real for center in centers])
    y_points = np.array([center.imag for center in centers])

    # Compute cumulative distance
    distances = np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Use cumulative distances as the parameter t
    t = cumulative_distances

    # Generate new t values for smooth interpolation
    t_new = np.linspace(t[0], t[-1], num_spline_points)

    # Determine the spline degree
    num_points = len(centers)
    if num_points < 2:
        print("Need at least two points to plot a spline.")
        return
    elif num_points < 4:
        k = num_points - 1
    else:
        k = 3  # Cubic spline

    # Create splines for x and y coordinates
    x_spline = make_interp_spline(t, x_points, k=k)
    y_spline = make_interp_spline(t, y_points, k=k)

    # Evaluate the splines to get smooth x and y values
    x_smooth = x_spline(t_new)
    y_smooth = y_spline(t_new)

    # Plotting
    fig, ax = plt.subplots()

    # Plot the spline
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Spline')

    # Plot the circles and their centers
    for center, radius in circles_data:
        circle = plt.Circle((center.real, center.imag), radius, fill=False, edgecolor='blue')
        ax.add_artist(circle)
        # Plot the center
        ax.plot(center.real, center.imag, 'ro', markersize=5)

    ax.set_aspect('equal', 'box')
    # Adjust plot limits
    all_centers = np.array(centers)
    min_x, max_x = np.min(all_centers.real), np.max(all_centers.real)
    min_y, max_y = np.min(all_centers.imag), np.max(all_centers.imag)
    padding = 1.0
    plt.xlim(min_x - padding, max_x + padding)
    plt.ylim(min_y - padding, max_y + padding)
    plt.grid(True)
    plt.title('Spline Through Knot Diagram Points')
    plt.legend()
    plt.show()

# End of Sequencing Spline ____________________________________________________
