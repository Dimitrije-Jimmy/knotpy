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
    koeff = np.sqrt(area / n)
    dt = 0.1          # Time step
    max_iterations = max_iterations
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

# End of SIM functions ________________________________________________________

# Start of SNAP functions _____________________________________________________

"""
Instructions for GPT


"""

def add_additional_points(ret):
    newret = ret.copy()

    for key, circle in ret.items():
        if isinstance(key, frozenset):
            id_from, id_to = key
            node_from = str(id_from)[0]
            node_to = str(id_to)[0]
            circles_from = ret[key] * ret[node_from]
            circles_to = ret[key] * ret[node_to]
            print(circles_from, circles_to)
            newret[(key, node_from)] = SimCircle(circles_from[0], 0.1)
            newret[(key, node_to)] = SimCircle(circles_to[0], 0.1)

    return newret

def remove_additional_points(ret):
    #newret = ret.copy()
    newret = copy.deepcopy(ret)


    for key, _ in ret.items():
        if isinstance(key, tuple):
            print(key)
            #del newret[key]
            newret.pop(key)

    return newret

def remove_additional_points(ret):
    #newret = ret.copy()
    newret2 = {}


    for key, circle in ret.items():
        if not isinstance(key, tuple):
            print(key)
            #del newret[key]
            #newret.pop(key)
            #newret[key] = circle
            newret2.update({key: circle})


    print("\nnewret2")
    pp.pprint(newret2)
    return newret2

def expand_added_circles(ret):
    print("Expanding circles...")
    #newret = ret.copy()
    newret = copy.deepcopy(ret)

    for key, circle in newret.items():
        if isinstance(key, tuple):
            id_from, id_to = key

            while (circle * ret[id_from]) == []:
                circle.radius += 0.01

            while (circle * ret[id_to]) == []:
                circle.radius += 0.01

    return newret

# Start circle fitting
"""

"""
def circle_from_three_points(p1, p2, p3):
    """
    Compute the circle passing through three non-colinear points.
    Returns the center (h, k) and radius r.
    """
    # Coordinates of the points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculate the determinants
    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2.0
    cd = (temp - x3**2 - y3**2) / 2.0
    det = (x1 - x2)*(y2 - y3) - (x2 - x3)*(y1 - y2)

    if abs(det) < 1e-10:
        raise ValueError("Points are colinear")

    # Center of circle (h, k)
    h = (bc*(y2 - y3) - cd*(y1 - y2)) / det
    k = ((x1 - x2)*cd - (x2 - x3)*bc) / det

    # Radius of circle
    r = np.sqrt((x1 - h)**2 + (y1 - k)**2)

    return h, k, r

def are_points_concyclic(points, tolerance=1e-6):
    """
    Check if all points are concyclic within a specified tolerance.
    Returns True if they are, False otherwise.
    """
    if len(points) <= 3:
        return True  # Three or fewer points are always concyclic

    # Compute circle from first three points
    h, k, r = circle_from_three_points(points[0], points[1], points[2])

    # Check if the rest of the points lie on this circle
    for p in points[3:]:
        x, y = p
        distance = np.sqrt((x - h)**2 + (y - k)**2)
        if abs(distance - r) > tolerance:
            return False
    return True

def circle_through_points(points):
    """
    Compute the circle passing through all given points.
    If the points are concyclic, returns the center and radius.
    Otherwise, raises a ValueError.
    """
    num_points = len(points)
    if num_points < 2:
        raise ValueError("At least two points are required")

    if num_points == 2:
        # Infinite circles pass through two points.
        # Return the circle with center at the midpoint and radius half the distance.
        x1, y1 = points[0]
        x2, y2 = points[1]
        h = (x1 + x2) / 2.0
        k = (y1 + y2) / 2.0
        r = np.sqrt((x1 - h)**2 + (y1 - k)**2)
        return h, k, r

    # For three or more points
    if are_points_concyclic(points):
        # Compute circle from first three points
        h, k, r = circle_from_three_points(points[0], points[1], points[2])
        return h, k, r
    else:
        raise ValueError("Points are not concyclic; no single circle passes through all points")


def fit_circle_least_squares(points):
    """
    Fit a circle to a set of points using least squares minimization.
    Returns the center (h, k) and radius r.
    """
    x = points[:, 0]
    y = points[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f_2, center_estimate)
    xc, yc = center
    Ri       = calc_R(xc, yc)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R


from scipy import optimize

def circle_through_points_or_fit(points):
    """
    Attempts to compute the circle passing through all given points.
    If not possible, fits a circle using least squares.
    Returns the center (h, k) and radius r.
    """
    try:
        # First, try to find an exact circle
        return circle_through_points(points)
    except ValueError:
        # If not possible, fit a circle
        print("Points are not concyclic. Fitting a circle using least squares.")
        h, k, r = fit_circle_least_squares(points)
        return h, k, r
# end circle fitting


def connecting_points(desired_key, ret):
    connection_keys = []
    for key, _ in ret.items():
        if isinstance(key, tuple):
            id_from, id_to = key
            if desired_key in key:
                connection_keys.append(ret[key])
    
    connections = []
    for circle in connection_keys:
        connections.append([circle.center.real, circle.center.imag])
   
    return connections

def run_snap(ret):
    #newret = ret.copy()
    newret = copy.deepcopy(ret)

    for key, _ in newret.items():
        if not isinstance(key, tuple):
            mandatory_points = connecting_points(key, ret)
            h, k, r = circle_through_points_or_fit(mandatory_points)
            #newret[key] = SimCircle(complex(h, k), r)
            newret[key].center = complex(h, k)
            newret[key].radius = r
    
    remove_additional_points(newret)
    return newret

# End of SNAP functions _______________________________________________________

if __name__ == "__main__":
    # Trefoil:
    s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
    # Large:
    #s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
    #s = "V[0,1,2], V[0,2,1]"
    #s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
    # Problematic:
    #s = "X[0,1,2,3],X[4,5,6,7],X[1,8,9,10],X[11,12,13,9],X[14,15,7,16],X[17,18,19,13],X[10,20,17,12],X[20,19,15,14],V[3,21,22],V[5,23,24],V[6,24,16],V[4,18,23],V[0,22,8],V[2,11,21]"
    k2 = from_pd_notation(s)
    print(k2)

    ret = circlepack_layout(k2)

    print("\n")
    pp.pprint(ret)
    print("\n")

    filtered_ret = extract_main_points_and_connections(ret)
    pp.pprint(filtered_ret)
    ret1 = add_additional_points(filtered_ret)
    #ret1 = remove_additional_points(ret1)
    #filtered_ret = ret
    #ret1 = run_sim(filtered_ret)
    #ret1 = {key: ret[key].center for key in ret}
    pp.pprint(ret1)
    ret2 = run_sim(ret1)
    pp.pprint(ret2)
    

    plot_circles(filtered_ret, 0)
    plot_circles(ret1, 0)
    plot_circles(ret2, 50)
    #ret3 = expand_added_circles(ret2)
    #plot_circles(ret3, 50)

    ret4 = run_snap(ret2)
    pp.pprint(ret4)
    plot_circles(ret4, 100)

    ret5 = remove_additional_points(ret4)
    ret5 = run_sim(ret5, 3)
    pp.pprint(ret5)
    plot_circles(ret5, 150)
    draw_from_layout(k2, ret5, with_labels=True, with_title=True)
    plt.plot()