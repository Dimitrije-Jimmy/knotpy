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

__all__ = ['SimCircle', 'extract_circle_positions', 'extract_main_points_and_connections', 'prep_sim', 'compute_repulsive_forces', 'limit_displacement', 'make_step', 'run_sim', 'plot_circles']
__version__ = 'god knows'


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
def compute_repulsive_forces(circles, koeff, dist_power=2):
    for _, ci in circles.items():
        ci.force = 0+0j  # Reset force
        for _, cj in circles.items():
            if ci != cj:
                delta = ci.center - cj.center
                distance = abs(delta) + 1e-6  # Avoid division by zero
                force_magnitude = koeff**2 / distance**(dist_power)
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

if __name__ == "__main__":
    from spline import plot_spline_normal, build_graph, build_circles_data, traverse_knot
    from spline import extract_unique_connections
    from spline import bezier_curve_matplotlib, bezier_curve_matplotlib_multiple

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

    print("\n")

    filtered_ret = extract_main_points_and_connections(ret)
    filtered_ret = ret
    #ret1 = run_sim(filtered_ret)
    ret1 = {key: ret[key].center for key in ret}
    pp.pprint(ret1)

    # Build graph
    graph, circles, identifiers = build_graph(ret1)

    # Extract unique connections
    connections = extract_unique_connections(graph)

    # Plot Bezier curves
    draw_from_layout(k2, ret, with_labels=True, with_title=True)
    bezier_curve_matplotlib_multiple(connections, circles)

    plot_circles(prep_sim(filtered_ret), 0)
    print("\nSimulation")
    ret1 = run_sim(filtered_ret)
    plot_circles(prep_sim(ret1), 50)
    ret2 = {key: ret1[key].center for key in ret1}
    pp.pprint(ret2)

    # Build graph
    graph, circles, identifiers = build_graph(ret2)

    # Extract unique connections
    connections = extract_unique_connections(graph)

    # Plot Bezier curves
    bezier_curve_matplotlib_multiple(connections, circles)

    # Plot Spline curves
    # Determine the sequence (adjust 'start_node' and traversal as needed)
    start_node = 'a'  # Adjust as appropriate
    sequence = traverse_knot(graph, start_node)

    # Build the ordered circles_data
    circles_data = build_circles_data(sequence, ret1)

    # Plot the spline
    plot_spline_normal(circles_data)

