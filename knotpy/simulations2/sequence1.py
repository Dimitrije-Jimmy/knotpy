"""
Simulation lower energy state, better looking knot
"""
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

#__all__ = ['draw', 'export_pdf', "circlepack_layout", "draw_from_layout", "add_support_arcs", "plt", "export_png"]
__version__ = 'god knows'


"""
Simple Trefoil:
PD notation: X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]

from_pd_notation -> PlanarDiagram object:
PlanarDiagram with 3 nodes, 6 arcs, and adjacencies a → X(c1 b0 b3 c2), b → X(a1 c0 c3 a2), c → X(b1 a0 a3 b2) with framing 0

circlepack_layout:
ret = {frozenset({b0, a1}): <knotpy.utils.geometry.Circle object at 0x000002B19BBB5CD0>,
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

Explanation of the ret dictionary:
- it contains circle objects with center and radius but they dont matter right now
- the keys are nodes represented with strings of a letter from a-z,
  the frozensets represent the arcs that connect the nodes, with the letter 
  of the node with a number on the side that also doesnt matter here except that 
  it needs to be present in the key of the ret dictionary

Simulation here:


draw_from_layout(circlepack_layout):
image


s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
PlanarDiagram with 5 nodes, 9 arcs, and adjacencies a → V(b0 d2 c3), b → V(a0 c2 e3), c → X(d1 e0 b1 a2), d → X(e1 c0 a1 e2), e → X(c1 d0 d3 b2) with framing 0

the function 'edges(k)' returns this sequence:
[[a0, b0], [a1, d2, d0, e1, e3, b2], [a2, c3, c1, e0, e2, d3, d1, c0, c2, b1]]
- starts counting sequence at random node
- each list is a sequence, if a knot has one sequence itll be one list,
  but if the knot is a theta diagram a.k.a. it has vertexes with 3 connections
  itll be multiple lists as it has to traverse multiple paths
- the function edges(k) returns this sequence, the exact thing I needed as a solution
  to my problem god damn! skoda da ni prej odgovoriu bi ze meu vec done
- there will always be atleast two elements within the list
- the elements are always in sequential order but the frozensets might have the opposite direction
  so what that means is that Im trying to match up these indeces within the sequence
  with the keys within the ret dictionary
  
I need a function that will take a sequence like the one above and return this:
[['a', frozenset({a0, b0}), 'b'], ['a', frozenset({a1, d2}), 'd', frozenset({d0, e1}), 'e', frozenset({e3, b2}) 'b'], 
 ['a', frozenset({a2, c3}), 'c', frozenset({c1, e0}), 'e', frozenset({e2, d3}), 'd', frozenset({d1, c0}), 'c', frozenset({c2, b1}) 'b']]

I need you to construct this function and take into account the things I wrote above
- make it modular by separating it into 2 functions, one should handle each list within
  the list on its own and the other the entire list just calling the other function
- lookup within the ret dictionary wether the frozenset is frozenset({a1, d2}) or frozenset({d2, a1})
  (check this for all the frozensets)

"""

# Simulation functions ________________________________________________________

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
    for key, circle in ret.items():
        #print(f"Key: {key}, Center: {circle.center}, Radius: {circle.radius}")
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
                force_magnitude = koeff**2 / distance**2
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

def run_sim(ret, max_iterations=100):
    """
    Run the simulation given a dictionary of circlepack layout ret.

    Parameters
    ----------
    ret : dict
        A dictionary of circlepack layout, where the keys are the node labels and
        the values are the Circle objects.

    Returns
    -------
    circles : dict
        A dictionary of Circle objects, where the keys are the node labels and
        the values are the Circle objects.

    Notes
    -----
    The simulation parameters are set as follows:
        - n: the number of circles
        - area: assumed unit area for simplicity
        - koeff: the repulsion constant, set to sqrt(area/n)
        - dt: the time step, set to 0.1
        - max_iterations: the maximum number of iterations, set to 50
        - temperature: the initial temperature to limit displacement, set to 0.1
    """
    circles = prep_sim(ret)

    # Simulation parameters
    n = len(circles)  # Number of circles
    area = 1.0        # Assume unit area for simplicity
    #area = 0.01        # Assume unit area for simplicity
    koeff = np.sqrt(area / n)
    dt = 0.1          # Time step
    max_iterations = max_iterations
    temperature = 0.01  # Initial temperature to limit displacement

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

# End of SIM functions ________________________________________________________

# Start of Seuqencing Spline __________________________________________________


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


def build_all_circles_data(sequences, circles):
    """
    Build circle data for multiple sequences.

    Parameters:
    - sequences: A list of sequences, where each sequence is a list of keys.
    - circles: A dictionary where keys are node or edge identifiers and values are circle objects.

    Returns:
    - A list where each element is the circles data for a sequence.
    """
    all_circles_data = []
    for sequence in sequences:
        circles_data = []
        for key in sequence:
            circle = circles.get(key)
            if circle:
                center = circle.center
                radius = circle.radius
                circles_data.append((center, radius))
            else:
                print(f"Warning: Circle not found for key {key}")
        all_circles_data.append(circles_data)
    return all_circles_data

def plot_splines(all_circles_data):
    from scipy.interpolate import make_interp_spline
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    all_centers = []

    # Define colors for each spline
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    color_cycle = iter(colors)

    for idx, circles_data in enumerate(all_circles_data):
        color = next(color_cycle, 'black')  # Use black if colors run out

        # Extract centers from circles_data
        centers = [center for center, _ in circles_data]
        x_points = np.array([center.real for center in centers])
        y_points = np.array([center.imag for center in centers])

        # Store all centers for plot limits
        all_centers.extend(centers)

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
            continue
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

        # Plot the spline
        ax.plot(x_smooth, y_smooth, color=color, linewidth=2, label=f'Spline {idx+1}')

        # Plot the circles and their centers
        for center, radius in circles_data:
            circle = plt.Circle((center.real, center.imag), radius, fill=False, edgecolor=color, alpha=0.5)
            ax.add_artist(circle)
            # Plot the center
            ax.plot(center.real, center.imag, 'o', color=color, markersize=5)

    ax.set_aspect('equal', 'box')
    # Adjust plot limits
    if all_centers:
        all_centers = np.array(all_centers)
        min_x, max_x = np.min(all_centers.real), np.max(all_centers.real)
        min_y, max_y = np.min(all_centers.imag), np.max(all_centers.imag)
        padding = 1.0
        plt.xlim(min_x - padding, max_x + padding)
        plt.ylim(min_y - padding, max_y + padding)
    plt.grid(True)
    plt.title('Splines Through Knot Diagram Points')
    plt.legend()
    plt.show()



# End of Sequencing Spline ____________________________________________________



# Start of Sequence functions _________________________________________________

def process_path(path, ret):
    """
    Process a single path to create a sequence of nodes and edges.

    Parameters:
    - path: A list of node identifiers (e.g., ['a1', 'd2', 'd0', 'e1', 'e3', 'b2']).
    - ret: A dictionary where keys are edges (frozensets or tuples of node identifiers) and values are circle objects.

    Returns:
    - A list containing the sequence of node labels and edges.
    """
    result = []
    node_labels = [str(node)[0] for node in path]  # Extract node labels (e.g., 'a1' -> 'a')
    current_label = node_labels[0]
    result.append(current_label)
    
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i+1]
        label1 = str(node1)[0]
        label2 = str(node2)[0]
        
        if label1 != label2:
            # Try to find the edge in ret
            edge = None
            edge_keys = [
                frozenset({node1, node2}),
                frozenset({node2, node1})
            ]
            for key in edge_keys:
                if key in ret:
                    edge = key
                    break
            if edge:
                #result.append(label1)
                result.append(edge)
                result.append(label2)
            else:
                raise KeyError(f"Edge between {node1} and {node2} not found in ret.")
        else:
            # Skip edges between nodes of the same label
            continue
    #print(result)
    return result

def process_paths(paths, ret):
    """
    Process multiple paths by applying process_path to each.

    Parameters:
    - paths: A list of paths, where each path is a list of node identifiers.
    - ret: A dictionary where keys are edges (frozensets or tuples of node identifiers) and values are circle objects.

    Returns:
    - A list of processed paths.
    """
    return [process_path(path, ret) for path in paths]


# End of Sequence functions ___________________________________________________



if __name__ == "__main__":
    # Trefoil:
    s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
    # Large:
    #s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
    #s = "V[0,1,2], V[0,2,1]"
    #s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
    # Problematic:
    s = "X[0,1,2,3],X[4,5,6,7],X[1,8,9,10],X[11,12,13,9],X[14,15,7,16],X[17,18,19,13],X[10,20,17,12],X[20,19,15,14],V[3,21,22],V[5,23,24],V[6,24,16],V[4,18,23],V[0,22,8],V[2,11,21]"
    k2 = from_pd_notation(s)
    print(k2)
    ret = circlepack_layout(k2)
    ret = extract_main_points_and_connections(ret)
    pp.pprint(ret)

    ret1 = run_sim(ret, 100)

    #draw_from_layout(k2, ret)
    #plt.show()
    #draw(k2, with_labels=True, with_title=True)
    #plt.show()

    print("\nPaths:")
    paths = edges(k2)
    print(paths)
    processed_paths = process_paths(paths, ret)
    print(processed_paths)

    # Build the ordered circles_data
    #circles_data = build_circles_data(processed_paths[0], ret)
    # Plot the spline
    #plot_spline(circles_data)

    # Build all circles data
    all_circles_data = build_all_circles_data(processed_paths, ret)
    all_circles_data2 = build_all_circles_data(processed_paths, ret1)
    # Plot all splines on the same figure
    plot_splines(all_circles_data)
    plot_splines(all_circles_data2)

    plt.show()


