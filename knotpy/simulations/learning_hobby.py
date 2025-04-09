from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pprint

pp = pprint.PrettyPrinter(indent=4)

# Imports and funcs for PD ________________________________________________________

from knotpy.drawing.circlepack import CirclePack
from knotpy.drawing.draw_matplotlib import draw_from_layout
from knotpy.notation.native import from_knotpy_notation
from knotpy.notation.pd import from_pd_notation
from knotpy.drawing.layout import circlepack_layout

def extract_circle_positions(circle_dict):
    """Extract circle centers from the dictionary of circles."""
    positions = {}
    for key, circle in circle_dict.items():
        positions[key] = circle.center  # Each circle has a .center attribute
    return positions

def extract_main_points_and_connections(ret):
    return {k: v for k, v in ret.items() if isinstance(k, (str, frozenset))}

# End of imports ________________________________________________________________

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

def extract_spline_points(connection, circles):
    """
    Extracts ordered points for the spline from a connection.
    """
    start_node, control_points_set, end_node = connection
    control_points = list(control_points_set)
    # Sort control points if necessary
    control_points.sort()  # Implement custom sorting if needed
    # Get positions
    points = [circles[start_node]]
    #for cp in control_points:
    #    points.append(circles[cp])
    points.append(circles[control_points_set])
    points.append(circles[end_node])
    pp.pprint(points)
    return points

def compute_hobby_tangents(points, tension=1.0):
    """
    Computes the tangents for Hobby's spline.
    """
    n = len(points)
    d = np.zeros(n - 1, dtype=complex)
    s = np.zeros(n - 1)
    tangents = np.zeros(n, dtype=complex)
    
    # Compute chord lengths and directions
    for i in range(n - 1):
        d[i] = points[i + 1] - points[i]
        s[i] = abs(d[i])
    
    # Compute unit vectors
    u = d / s[:, np.newaxis]
    
    # Compute tangents
    for i in range(1, n - 1):
        alpha = s[i - 1] / (s[i - 1] + s[i])
        tangents[i] = (1 - alpha) * tension * d[i - 1] + alpha * tension * d[i]
    
    # Endpoints tangents (natural spline conditions)
    tangents[0] = tension * d[0]
    tangents[-1] = tension * d[-1]
    
    return tangents

def evaluate_hobby_spline(points, tangents, num=100):
    """
    Evaluates the Hobby spline at multiple points.
    """
    n = len(points)
    curve_x = []
    curve_y = []
    
    for i in range(n - 1):
        # Hermite cubic coefficients
        p0 = points[i]
        p1 = points[i + 1]
        m0 = tangents[i]
        m1 = tangents[i + 1]
        
        t = np.linspace(0, 1, num)
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        
        curve_segment = (h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
        curve_x.extend(curve_segment.real)
        curve_y.extend(curve_segment.imag)
    
    return np.array(curve_x), np.array(curve_y)

def plot_hobby_splines_multiple(connections, circles):
    """Plots multiple Hobby splines on the same plot."""
    plt.figure(figsize=(10, 8))
    
    for connection in connections:
        points = extract_spline_points(connection, circles)
        tangents = compute_hobby_tangents(points)
        curve_x, curve_y = evaluate_hobby_spline(points, tangents)
        # Plot the spline
        plt.plot(curve_x, curve_y, 'b-', linewidth=2)
        # Plot control points
        x_points = [p.real for p in points]
        y_points = [p.imag for p in points]
        #plt.plot(x_points, y_points, 'ro--', markersize=4)
    
    plt.title("Hobby Splines for Graph Connections")
    plt.grid(True)
    plt.axis('equal')
    plt.show()


s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
#s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
#s = "V[0,1,2], V[0,2,1]"
#s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
# problematic
#s = "X[0,1,2,3],X[4,5,6,7],X[1,8,9,10],X[11,12,13,9],X[14,15,7,16],X[17,18,19,13],X[10,20,17,12],X[20,19,15,14],V[3,21,22],V[5,23,24],V[6,24,16],V[4,18,23],V[0,22,8],V[2,11,21]"
k2 = from_pd_notation(s)
print(k2)

ret = circlepack_layout(k2)

print("\n")

filtered_ret = extract_main_points_and_connections(ret)
#ret1 = simulation(filtered_ret)
ret = filtered_ret
ret1 = {key: ret[key].center for key in ret}
pp.pprint(ret1)

# Build graph
graph, circles, identifiers = build_graph(ret)

# Extract unique connections
connections = extract_unique_connections(graph)
#pp.pprint(connections)
circles = extract_circle_positions(circles)
#pp.pprint(circles)
# Plot Hobby splines
plot_hobby_splines_multiple(connections, circles)



