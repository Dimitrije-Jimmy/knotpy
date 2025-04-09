"""
Simulation lower energy state, better looking knot
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint as pp

from knotpy.notation.native import from_knotpy_notation, from_pd_notation
from knotpy.notation.pd import to_pd_notation
from knotpy.drawing.draw_matplotlib import draw, draw_from_layout
from knotpy.utils.geometry import Circle

import knotpy as kp
from knotpy.drawing.layout import circlepack_layout

#__all__ = ['draw', 'export_pdf', "circlepack_layout", "draw_from_layout", "add_support_arcs", "plt", "export_png"]
__version__ = '0.3.5 - Splines'
__author__ = 'Dimitrije Pešić'


"""
Simple Trefoil:
PD notation: X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]

from_pd_notation -> PlanarDiagram object:
PlanarDiagram with 3 nodes, 6 arcs, and adjacencies a → X(c1 b0 b3 c2), b → X(a1 c0 c3 a2), c → X(b1 a0 a3 b2) with framing 0

circlepack_layout:
{frozenset({b0, a1}): <knotpy.utils.geometry.Circle object at 0x000002B19BBB5CD0>,
 frozenset({c1, a0}): <knotpy.utils.geometry.Circle object at 0x000002B19B40BCB0>,
 frozenset({c0, b1}): <knotpy.utils.geometry.Circle object at 0x000002B19CC01AC0>,
 frozenset({a2, b3}): <knotpy.utils.geometry.Circle object at 0x000002B19CC560F0>,
 frozenset({a3, c2}): <knotpy.utils.geometry.Circle object at 0x000002B19CC56120>,
 frozenset({c3, b2}): <knotpy.utils.geometry.Circle object at 0x000002B19CC56150>,
 'a': <knotpy.utils.geometry.Circle object at 0x000002B19B3587D0>,
 'b': <knotpy.utils.geometry.Circle object at 0x000002B19AC882F0>,
 'c': <knotpy.utils.geometry.Circle object at 0x000002B19B3585F0>,
 (b0, a2): <knotpy.utils.geometry.Circle object at 0x000002B19CC2D340>,
 (b2, c0): <knotpy.utils.geometry.Circle object at 0x000002B19CC561B0>,
 (c2, a0): <knotpy.utils.geometry.Circle object at 0x000002B19CC56180>,
 (c3, a3, b3): <knotpy.utils.geometry.Circle object at 0x000002B19CC2D520>}

Simulation here:


draw_from_layout(circlepack_layout):
image
"""


def extract_circle_positions(circle_dict):
    """Extract circle centers from the dictionary of circles."""
    positions = {}
    for key, circle in circle_dict.items():
        positions[key] = circle.center  # Each circle has a .center attribute
    return positions

def extract_main_points_and_connections(ret):
    return {k: v for k, v in ret.items() if isinstance(k, (str, frozenset))}


class SimCircle(Circle):
    def __init__(self, center, radius):
        super().__init__(center, radius)
        self.force = 0+0j      # Initialize force as a complex number
        self.velocity = 0+0j   # Initialize velocity as a complex number
        self.mass = 1.0        # Assume unit mass for simplicity
        self.positions_ot = []
        self.radius_ot = []

def prep_sim(ret): 
    circlepack_layout_sim = {}
    print("\n, prep sim")
    for key, circle in ret.items():
        print(f"Key: {key}, Center: {circle.center}, Radius: {circle.radius}")
        c = circle.center
        r = circle.radius
        sim_circle = SimCircle(c, r)
        circlepack_layout_sim[key] = sim_circle
    
    return circlepack_layout_sim



# Function to compute repulsive forces
def compute_repulsive_forces(circles, koeff):
    for _, ci in circles.items():
        ci.force = 0+0j  # Reset force
        for _, cj in circles.items():
            if ci != cj:
                delta = ci.center - cj.center
                distance = abs(delta) + 1e-6  # Avoid division by zero
                force_magnitude = koeff**2 / distance
                force_direction = delta / distance
                ci.force += force_magnitude * force_direction


def limit_displacement(delta, temperature):
    delta_magnitude = abs(delta)
    if delta_magnitude > temperature:
        return delta / delta_magnitude * temperature
    else:
        return delta


def make_step(circles, koeff, dt, iteration, temperature):

    compute_repulsive_forces(circles, koeff)

    # Update positions
    for key, circle in circles.items():
        displacement = circle.force * dt
        displacement = limit_displacement(displacement, temperature)
        circle.center += displacement

        # Store positions for plotting every 10 iterations
        if iteration % 10 == 0:
            circle.positions_ot.append(circle.center)
            circle.radius_ot.append(circle.radius)

    # Cool down the system
    temperature *= 0.95  # Decrease temperature over time

def run_sim(ret):
    circles = prep_sim(ret)

    # Simulation parameters
    n = len(circles)  # Number of circles
    area = 1.0        # Assume unit area for simplicity
    koeff = np.sqrt(area / n)
    dt = 0.1          # Time step
    #max_iterations = max_iterations
    max_iterations = 50
    temperature = 0.1  # Initial temperature to limit displacement

    # Simulation loop
    for iteration in range(max_iterations):
        make_step(circles, koeff, dt, iteration, temperature) 
        
    return circles

# Visualization function
def plot_circles(circles_data, iteration):
    fig, ax = plt.subplots(figsize=(6,6))

    for key, circle_obj in circles_data.items():
        center = circle_obj.center
        radius = circle_obj.radius
        circle = plt.Circle((center.real, center.imag), radius, fill=False, edgecolor='blue')
        ax.add_artist(circle)
        # Plot the center
        ax.plot(center.real, center.imag, 'ro', markersize=2)
        ax.text(center.real, center.imag, key, fontsize=8)
    
    ax.set_aspect('equal', 'box')
    ax.set_title(f'Iteration {iteration}')

    # Adjust plot limits
    all_centers = np.array([circle.center for _, circle in circles_data.items()])
    min_x, max_x = np.min(all_centers.real), np.max(all_centers.real)
    min_y, max_y = np.min(all_centers.imag), np.max(all_centers.imag)
    padding = 1.0
    plt.xlim(min_x - padding, max_x + padding)
    plt.ylim(min_y - padding, max_y + padding)
    plt.grid(True)

    plt.show()


s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
#s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
##s = "V[0,1,2], V[0,2,1]"
##s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
#s = "V[2,6,1], V[7,1,8], V[3,8,4], V[5,4,6], X[5,2,7,3]"
#
#s = "('PlanarDiagram', {'name': '+t3_1#-3_1(1).2'}, [('Vertex', 'a', (('IngoingEndpoint', 'c', 2, {}), ('OutgoingEndpoint', 'b', 1, {'color': 1}), ('OutgoingEndpoint', 'd', 0, {})), {}), ('Vertex', 'b', (('IngoingEndpoint', 'd', 1, {}), ('IngoingEndpoint', 'a', 1, {'color': 1}), ('OutgoingEndpoint', 'g', 0, {})), {}), ('Crossing', 'c', (('IngoingEndpoint', 'f', 3, {}), ('IngoingEndpoint', 'f', 2, {}), ('OutgoingEndpoint', 'a', 0, {}), ('OutgoingEndpoint', 'e', 0, {})), {}), ('Crossing', 'd', (('IngoingEndpoint', 'a', 2, {}), ('OutgoingEndpoint', 'b', 0, {}), ('OutgoingEndpoint', 'g', 3, {}), ('IngoingEndpoint', 'h', 2, {})), {}), ('Crossing', 'e', (('IngoingEndpoint', 'c', 3, {}), ('IngoingEndpoint', 'h', 1, {}), ('OutgoingEndpoint', 'f', 1, {}), ('OutgoingEndpoint', 'f', 0, {})), {}), ('Crossing', 'f', (('IngoingEndpoint', 'e', 3, {}), ('IngoingEndpoint', 'e', 2, {}), ('OutgoingEndpoint', 'c', 1, {}), ('OutgoingEndpoint', 'c', 0, {})), {}), ('Crossing', 'g', (('IngoingEndpoint', 'b', 2, {}), ('OutgoingEndpoint', 'h', 0, {}), ('OutgoingEndpoint', 'h', 3, {}), ('IngoingEndpoint', 'd', 2, {})), {}), ('Crossing', 'h', (('IngoingEndpoint', 'g', 1, {}), ('OutgoingEndpoint', 'e', 1, {}), ('OutgoingEndpoint', 'd', 3, {}), ('IngoingEndpoint', 'g', 2, {})), {})])"
#k = from_knotpy_notation(s)
#s = "V[0,1,2],V[3,1,4],X[5,6,0,7],X[2,3,8,9],X[7,10,11,12],X[12,11,6,5],X[4,13,14,8],X[13,10,9,14]"
print("exiting spline6")
"""
if __name__ == "__name__":
    print("knotpy/simulations/spline6_411Eulerian2.py")
    exit()
exit()
"""
print("\n")
print(s)
k2 = from_pd_notation(s)
#k2 = k
print(k2)


# always named ret
ret = circlepack_layout(k2)

print("\n")
pp.pprint(ret)
print("\n")
filtered_ret = extract_main_points_and_connections(ret)
pp.pprint(filtered_ret)

#plot_circles(filtered_ret, 'filtered ret')
draw_from_layout(k2, filtered_ret, with_labels=True, with_title=True)

print("\n")

plot_circles(prep_sim(ret), 0)
#positions_over_time = run_sim(ret)
positions_over_time = run_sim(filtered_ret)
#pp.pprint(positions_over_time[0])
#pp.pprint(positions_over_time[0])

# Plotting the positions
#for idx, circles_data in enumerate(positions_over_time):
#    iteration = idx * 10
#    plot_circles(circles_data, iteration)

#print(len(positions_over_time))
pp.pprint(positions_over_time)
plot_circles(positions_over_time, 100)



#pp.pprint(list(circlepack_layout_sim.values()) == positions_over_time[0])

#draw_from_layout(k2, circlepack_layout_sim, with_labels=True, with_title=True)


#draw_from_layout(k2, ret, with_labels=True, with_title=True)
#plt.show()
exit()
# Attemp ___________________________________________________

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

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
    print("Knot Traversal: ", sequence)
    return sequence

# Eulerian Sequence _______________________________________________________ 

from collections import defaultdict
import copy

def find_eulerian_circuit(graph, start_node):
    # Copy the graph to track unused edges
    unused_edges = {node: edges.copy() for node, edges in graph.items()}
    stack = [(start_node, None)]  # Stack stores tuples of (node, edge_label)
    circuit = []

    while stack:
        current_node, edge_to_current = stack[-1]
        if unused_edges[current_node]:
            # Get the next unused edge
            neighbor_node, edge_label = unused_edges[current_node].pop()
            # Remove the corresponding edge from the neighbor's list
            for idx, (nbr, lbl) in enumerate(unused_edges[neighbor_node]):
                if nbr == current_node and lbl == edge_label:
                    unused_edges[neighbor_node].pop(idx)
                    break
            # Push the neighbor onto the stack
            stack.append((neighbor_node, edge_label))
        else:
            # No unused edges left from current_node, backtrack
            circuit.append(stack.pop())

    # Reverse the circuit to get the correct order
    circuit = circuit[::-1]

    # Build the sequence
    sequence = []
    for idx, (node, edge_label) in enumerate(circuit):
        if idx == 0:
            sequence.append(node)
        else:
            sequence.append(edge_label)
            sequence.append(node)

    return sequence


graph, circles, identifiers = build_graph(ret)
sequence = find_eulerian_circuit(graph, 'a')
print("Eulerian: ", sequence)


# End of attempt ________________________________________________________


def build_circles_data(sequence, circles):
    circles_data = []
    for key in sequence:
        circle = circles.get(key)
        if circle:
            center = circle.center
            radius = circle.radius
            circles_data.append((center, radius))
        else:
            print(f"Warning: Circle not found for key {key}")
    return circles_data


def plot_spline(circles_data, sequence):
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
    t_new = np.linspace(t[0], t[-1], 300)

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
    for i, data in enumerate(circles_data):
        center, radius = data
        circle = plt.Circle((center.real, center.imag), radius, fill=False, edgecolor='blue')
        ax.add_artist(circle)
        # Plot the center
        ax.plot(center.real, center.imag, 'ro', markersize=5)

        ax.text(center.real+0.1, center.imag+0.1, sequence[i], ha='center', va='center')

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
    #plt.show()


# Bezier implementation attempt ____________________________________________

def build_points_from_sequence(sequence, circles):
    points = []
    for key in sequence:
        circle = circles.get(key)
        if circle:
            points.append(circle.center)
        else:
            print(f"Warning: Circle not found for key {key}")
    return points

def compute_bezier_control_points(points):
    n = len(points) - 1  # Number of segments
    P1 = [None] * n
    P2 = [None] * n

    # Right-hand side vector
    rhs = []

    # Set up the equations for control points
    for i in range(n):
        if i == 0:
            # First segment
            rhs.append(points[0] + 2 * points[1])
        elif i == n - 1:
            # Last segment
            rhs.append(8 * points[n - 1] + points[n])
        else:
            # Middle segments
            rhs.append(4 * points[i] + 2 * points[i + 1])

    # Build the tridiagonal system
    a = [0] + [1] * (n - 1)
    b = [2] + [4] * (n - 2) + [7]
    c = [1] * (n - 1) + [0]

    # Solve the tridiagonal system
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        rhs[i] = rhs[i] - m * rhs[i - 1]

    # Back substitution
    P1[n - 1] = rhs[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        P1[i] = (rhs[i] - c[i] * P1[i + 1]) / b[i]

    # Compute P2
    for i in range(n - 1):
        P2[i] = 2 * points[i + 1] - P1[i + 1]
    #P2.append((P1[n - 1] + points[n]) / 2)
    # Assign P2[n - 1]
    P2[n - 1] = (P1[n - 1] + points[n]) / 2

    # Print control points
    #pp.pprint(P1)
    #pp.pprint(P2)
    return P1, P2


import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def plot_cubic_bezier_from_points(points, sequence):
    P1, P2 = compute_bezier_control_points(points)

    # Build the path
    vertices = []
    codes = []

    # Start at the first point
    vertices.append((points[0].real, points[0].imag))
    codes.append(Path.MOVETO)

    for i in range(len(P1)):
        # Add control points and end point
        cp1 = P1[i]
        cp2 = P2[i]
        #print(cp2, cp2)
        p = points[i + 1]
        vertices.extend([
            (cp1.real, cp1.imag),
            (cp2.real, cp2.imag),
            (p.real, p.imag)
        ])
        codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

    # Create the Path object
    path = Path(vertices, codes)
    patch = PathPatch(path, facecolor='none', edgecolor='red', linewidth=2)

    # Plotting
    fig, ax = plt.subplots()
    ax.add_patch(patch)

    # Plot the original points
    x_points = [z.real for z in points]
    y_points = [z.imag for z in points]
    ax.plot(x_points, y_points, 'o', color='blue')

    for i, text in enumerate(sequence):
        ax.text(x_points[i]+0.1, y_points[i]+0.1, text, ha='center', va='center')

    ax.set_aspect('equal', 'box')
    plt.title('Knot Diagram with Bezier Curves')
    plt.grid(True)
    #plt.show()


def nekplotting(graph, circles, identifiers):
    import math
    # Now, assign positions to all identifiers
    identifier_positions = {}

    node_positions = {
    'a': circles['a'].center,
    'b': circles['b'].center,
    'c': circles['c'].center  # Position to form an equilateral triangle
        }

    # First, assign positions to main nodes
    for node in ['a', 'b', 'c']:
        identifier_positions[node] = node_positions[node]

    # Then, distribute sub-identifiers around their main nodes
    sub_identifier_angles = defaultdict(int)  # Keep track of angles used

    for identifier in identifiers:
        if identifier not in identifier_positions:
            main_node = identifier[0]
            angle = sub_identifier_angles[main_node]
            angle_rad = math.radians(angle)
            distance = 2  # Distance from the main node
            main_position = identifier_positions[main_node]
            # Calculate position
            x = main_position[0] + distance * math.cos(angle_rad)
            y = main_position[1] + distance * math.sin(angle_rad)
            identifier_positions[identifier] = np.array([x, y])
            # Update angle for next sub-identifier
            sub_identifier_angles[main_node] += 45  # Increment angle by 45 degrees

    # Now, plot the circles and connections
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot circles and labels
    for identifier, position in identifier_positions.items():
        circle = plt.Circle(position, 0.5, fill=True, color='skyblue', zorder=2)
        ax.add_patch(circle)
        plt.text(position[0], position[1], identifier, fontsize=9, ha='center', va='center', zorder=3)

    # Draw Bezier curves between connected identifiers
    from matplotlib.path import Path
    import matplotlib.patches as patches

    for key in circles:
        if isinstance(key, frozenset):
            id_from, id_to = key
            pos_from = identifier_positions[id_from]
            pos_to = identifier_positions[id_to]
            # Calculate control points for the Bezier curve
            ctrl1 = pos_from + (pos_to - pos_from) / 3 + np.array([0, 1])  # Offset for curve
            ctrl2 = pos_from + 2 * (pos_to - pos_from) / 3 + np.array([0, -1])
            # Create the Bezier path
            verts = [
                (pos_from[0], pos_from[1]),  # P0
                (ctrl1[0], ctrl1[1]),        # P1
                (ctrl2[0], ctrl2[1]),        # P2
                (pos_to[0], pos_to[1]),      # P3
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor='blue', lw=1, zorder=1)
            ax.add_patch(patch)
            # Optionally, label the connection
            mid_point = (pos_from + pos_to) / 2
            plt.text(mid_point[0], mid_point[1], f"{id_from}-{id_to}", fontsize=7, color='red', zorder=4)

    # Set plot limits
    plt.xlim(-5, 15)
    plt.ylim(-5, 15)
    plt.axis('equal')
    plt.axis('off')
    plt.show()


# End of Bezier implement attempt __________________________________________ 

# Example usage:

# Build the graph and circles mapping
graph, circles, identifiers = build_graph(positions_over_time)
#graph, circles, identifiers = build_graph(filtered_ret)

# Determine the sequence (adjust 'start_node' and traversal as needed)
start_node = 'a'  # Adjust as appropriate
sequence = traverse_knot(graph, start_node)

# Build the ordered circles_data
circles_data = build_circles_data(sequence, circles)

# Plot the spline
plot_spline(circles_data, sequence)
#points = build_points_from_sequence(sequence, circles)
#plot_cubic_bezier_from_points(points, sequence)

#nekplotting(graph, circles, identifiers)

# I want to try and plot only a few points that are in order
partial_ret = {}
for i in range(3):
    partial_ret[sequence[i]] = ret[sequence[i]]

print('\nCan I plot only a few circles.')
print(k2)
pp.pprint(partial_ret)
draw_from_layout(k2, partial_ret)

plt.show()



# End of Attempt __________________________________________

exit()

if __name__ == "__main__":
    # test
    s = "('PlanarDiagram', {'name': '+t3_1#-3_1(1).2'}, [('Vertex', 'a', (('IngoingEndpoint', 'c', 2, {}), ('OutgoingEndpoint', 'b', 1, {'color': 1}), ('OutgoingEndpoint', 'd', 0, {})), {}), ('Vertex', 'b', (('IngoingEndpoint', 'd', 1, {}), ('IngoingEndpoint', 'a', 1, {'color': 1}), ('OutgoingEndpoint', 'g', 0, {})), {}), ('Crossing', 'c', (('IngoingEndpoint', 'f', 3, {}), ('IngoingEndpoint', 'f', 2, {}), ('OutgoingEndpoint', 'a', 0, {}), ('OutgoingEndpoint', 'e', 0, {})), {}), ('Crossing', 'd', (('IngoingEndpoint', 'a', 2, {}), ('OutgoingEndpoint', 'b', 0, {}), ('OutgoingEndpoint', 'g', 3, {}), ('IngoingEndpoint', 'h', 2, {})), {}), ('Crossing', 'e', (('IngoingEndpoint', 'c', 3, {}), ('IngoingEndpoint', 'h', 1, {}), ('OutgoingEndpoint', 'f', 1, {}), ('OutgoingEndpoint', 'f', 0, {})), {}), ('Crossing', 'f', (('IngoingEndpoint', 'e', 3, {}), ('IngoingEndpoint', 'e', 2, {}), ('OutgoingEndpoint', 'c', 1, {}), ('OutgoingEndpoint', 'c', 0, {})), {}), ('Crossing', 'g', (('IngoingEndpoint', 'b', 2, {}), ('OutgoingEndpoint', 'h', 0, {}), ('OutgoingEndpoint', 'h', 3, {}), ('IngoingEndpoint', 'd', 2, {})), {}), ('Crossing', 'h', (('IngoingEndpoint', 'g', 1, {}), ('OutgoingEndpoint', 'e', 1, {}), ('OutgoingEndpoint', 'd', 3, {}), ('IngoingEndpoint', 'g', 2, {})), {})])"
    k = from_knotpy_notation(s)
    s = "V[0,1,2],V[3,1,4],X[5,6,0,7],X[2,3,8,9],X[7,10,11,12],X[12,11,6,5],X[4,13,14,8],X[13,10,9,14]"
    
    s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
    #s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
    s = "V[0,1,2], V[0,2,1]"
    s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
    print("\n")
    print(s)
    k2 = from_pd_notation(s)
    print(k2)
 
    ret = circlepack_layout(k2)

    print("\n")
    pp.pprint(ret)


    draw_from_layout(k2, ret, with_labels=True, with_title=True)
    
    # Example usage:
    circle_positions = extract_circle_positions(ret)
    for circle, position in circle_positions.items():
        print(f"Circle {circle}: Position {position}")

    draw_from_layout(k2, ret, with_labels=True, with_title=True)
    plt.show()