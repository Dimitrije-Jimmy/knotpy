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
    max_iterations = 100
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

def order_circles(circles_data):
    # Extract centers from circles_data in order of knot


    centers = [circle.center for _, circle in circles_data.items()]

    centers = [center for center, _ in circles_data]
   

    return centers

def plot_spline(circles_data):
    from scipy.interpolate import make_interp_spline

    # Extract centers from circles_data
    centers = [center for center, _ in circles_data]
    x_points = [center.real for center in centers]
    y_points = [center.imag for center in centers]

    # Create a parameter t based on the number of points
    t = np.arange(len(centers))

    # Generate new t values for smooth interpolation
    t_new = np.linspace(t[0], t[-1], 300)

    # Create splines for x and y coordinates
    x_spline = make_interp_spline(t, x_points, k=3)  # Cubic spline
    y_spline = make_interp_spline(t, y_points, k=3)

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



s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
#s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
##s = "V[0,1,2], V[0,2,1]"
##s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
#s = "V[2,6,1], V[7,1,8], V[3,8,4], V[5,4,6], X[5,2,7,3]"
#
#s = "('PlanarDiagram', {'name': '+t3_1#-3_1(1).2'}, [('Vertex', 'a', (('IngoingEndpoint', 'c', 2, {}), ('OutgoingEndpoint', 'b', 1, {'color': 1}), ('OutgoingEndpoint', 'd', 0, {})), {}), ('Vertex', 'b', (('IngoingEndpoint', 'd', 1, {}), ('IngoingEndpoint', 'a', 1, {'color': 1}), ('OutgoingEndpoint', 'g', 0, {})), {}), ('Crossing', 'c', (('IngoingEndpoint', 'f', 3, {}), ('IngoingEndpoint', 'f', 2, {}), ('OutgoingEndpoint', 'a', 0, {}), ('OutgoingEndpoint', 'e', 0, {})), {}), ('Crossing', 'd', (('IngoingEndpoint', 'a', 2, {}), ('OutgoingEndpoint', 'b', 0, {}), ('OutgoingEndpoint', 'g', 3, {}), ('IngoingEndpoint', 'h', 2, {})), {}), ('Crossing', 'e', (('IngoingEndpoint', 'c', 3, {}), ('IngoingEndpoint', 'h', 1, {}), ('OutgoingEndpoint', 'f', 1, {}), ('OutgoingEndpoint', 'f', 0, {})), {}), ('Crossing', 'f', (('IngoingEndpoint', 'e', 3, {}), ('IngoingEndpoint', 'e', 2, {}), ('OutgoingEndpoint', 'c', 1, {}), ('OutgoingEndpoint', 'c', 0, {})), {}), ('Crossing', 'g', (('IngoingEndpoint', 'b', 2, {}), ('OutgoingEndpoint', 'h', 0, {}), ('OutgoingEndpoint', 'h', 3, {}), ('IngoingEndpoint', 'd', 2, {})), {}), ('Crossing', 'h', (('IngoingEndpoint', 'g', 1, {}), ('OutgoingEndpoint', 'e', 1, {}), ('OutgoingEndpoint', 'd', 3, {}), ('IngoingEndpoint', 'g', 2, {})), {})])"
#k = from_knotpy_notation(s)
#s = "V[0,1,2],V[3,1,4],X[5,6,0,7],X[2,3,8,9],X[7,10,11,12],X[12,11,6,5],X[4,13,14,8],X[13,10,9,14]"


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

# Attemp ___________________________________________________

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Assuming SimCircle is already defined
print("\n start of new attempt \n")
"""def build_graph(ret):
    graph = defaultdict(list)
    circles = {}
    identifiers = set()

    for key, circle in ret.items():
        if isinstance(key, frozenset):
            # Key is a tuple representing a connection from key[0] to key[1]
            #print(key)
            id_from, id_to = key
            node_from = str(id_from)[0]
            node_to = str(id_to)[0]
            graph[node_from].append((node_to, key))  # Store the connection key
            # Map the connection to its circle
            circles[key] = circle
            identifiers.update([id_from, id_to])
        elif isinstance(key, str):
            # Key is a node
            print(key)
            node = str(key)
            circles[node] = circle
            identifiers.add(node)
    
    pp.pprint(graph)
    #pp.pprint(circles)

    return graph, circles"""
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

    return graph, circles


"""def traverse_knot(graph, start_node):
    visited_edges = set()
    sequence = []

    def dfs(node):
        sequence.append(node)
        for neighbor, conn_key in graph[node]:
            edge = (node, neighbor, conn_key)
            if edge not in visited_edges:
                visited_edges.add(edge)
                sequence.append(conn_key)  # Add the connection
                dfs(neighbor)
                # Optional: sequence.append(node)  # If you want to record returning to the node

    dfs(start_node)
    print(sequence)
    return sequence"""
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
        if circle:
            center = circle.center
            radius = circle.radius
            circles_data.append((center, radius))
        else:
            print(f"Warning: Circle not found for key {key}")
    return circles_data

def plot_spline(circles_data):
    from scipy.interpolate import make_interp_spline

    # Extract centers from circles_data
    centers = [center for center, _ in circles_data]
    x_points = [center.real for center in centers]
    y_points = [center.imag for center in centers]

    # Create a parameter t based on the number of points
    t = np.arange(len(centers))

    # Generate new t values for smooth interpolation
    t_new = np.linspace(t[0], t[-1], 150)

    # Create splines for x and y coordinates
    x_spline = make_interp_spline(t, x_points, k=3)  # Cubic spline
    y_spline = make_interp_spline(t, y_points, k=3)

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

# Example usage:

# Build the graph and circles mapping
graph, circles = build_graph(positions_over_time)
#graph, circles = build_graph(filtered_ret)

# Determine the sequence (adjust 'start_node' and traversal as needed)
start_node = 'a'  # Adjust as appropriate
sequence = traverse_knot(graph, start_node)

# Build the ordered circles_data
circles_data = build_circles_data(sequence, circles)

# Plot the spline
plot_spline(circles_data)


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