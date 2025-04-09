"""
Main Dev script, utilising subsidiaries
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

from simulation import SimCircle, extract_circle_positions, extract_main_points_and_connections
#from simulation import prep_sim, compute_repulsive_forces, limit_displacement, make_step, run_sim, plot_circles

from spline import plot_spline_normal, build_circles_data, extract_points
from spline import plot_bezier_curve_matplotlib,bezier_curve_matplotlib, bezier_curve_matplotlib_multiple, get_positions

from circlesnap import add_additional_points, remove_additional_points, run_snap

from sequence import process_paths, process_path#, plot_splines, build_all_circles_data
from over_under_pass import plot_splines, build_all_circles_data, build_skip_indices, build_skip_indices_for_all_paths

from circlesnap import add_additional_points, remove_additional_points

__all__ = ['run_complex_sim']
#__all__ = ['process_paths', 'process_path', 'plot_splines', 'build_all_circles_data']
__version__ = 'god knows'
__author__ = 'Dimitrije Pešić'



"""
Simple Trefoil:
PD notation: X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]

k2 = from_pd_notation() -> PlanarDiagram object:
PlanarDiagram with 3 nodes, 6 arcs, and adjacencies a → X(c1 b0 b3 c2), b → X(a1 c0 c3 a2), c → X(b1 a0 a3 b2) with framing 0

circlepack_layout(k2):
ret = {frozenset({b0, a1}): <SimCircle object at 0x000002B19BBB5CD0>,
 frozenset({c1, a0}): <SimCircle object at 0x000002B19B40BCB0>,
 frozenset({c0, b1}): <SimCircle object at 0x000002B19CC01AC0>,
 frozenset({a2, b3}): <SimCircle object at 0x000002B19CC560F0>,
 frozenset({a3, c2}): <SimCircle object at 0x000002B19CC56120>,
 frozenset({c3, b2}): <SimCircle object at 0x000002B19CC56150>,
 'a': <SimCircle object at 0x000002B19B3587D0>,
 'b': <SimCircle object at 0x000002B19AC882F0>,
 'c': <SimCircle object at 0x000002B19B3585F0>,
 (b0, a2): <SimCircle object at 0x000002B19CC2D340>,
 (b2, c0): <SimCircle object at 0x000002B19CC561B0>,
 (c2, a0): <SimCircle object at 0x000002B19CC56180>,
 (c3, a3, b3): <SimCircle object at 0x000002B19CC2D520>}

Explanation of the ret dictionary:
- it contains circle objects with center and radius but they dont matter right now
- the keys are nodes represented with strings of a letter from a-z,
  the frozensets represent the arcs that connect the nodes, with the letter 
  of the node with a number on the side that also doesnt matter here except that 
  it needs to be present in the key of the ret dictionary
 
class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __contains__(self, point):
        # Does the point lie on the circle?
        #print(abs(abs(point - self.center) - self.radius))
        return abs(abs(point - self.center) - self.radius) <= DIAMETER_ERROR

    def __mul__(self, other):
        # Intersection between geometric objects.


        if isinstance(other, Circle):
            return _intersection_circle_circle(self, other)

        if isinstance(other, Line):
            return _intersection_circle_line(self, other)

        raise TypeError(f"Intersection of a circle and {type(other)} not supported")

    def length(self):
        return 2 * math.pi * self.radius  # 2pi*r

    def __call__(self, angle1, angle2=None):
        # The point line at angle or the circular arc if two angles are give.
        if angle2 is not None:
            return CircularArc(self.center, self.radius, angle1, angle2)
        raise NotImplementedError()

    def __str__(self):
        return f"Circle at {self.center:.5f} with radius {self.radius:.5f}"


class SimCircle(Circle):
    def __init__(self, center, radius):
        super().__init__(center, radius)
        self.force = 0+0j      # Initialize force as a complex number
        self.velocity = 0+0j   # Initialize velocity as a complex number
        self.mass = 1.0        # Assume unit mass for simplicity
        self.positions_ot = []
        self.radius_ot = []

print("\nPaths:")
paths = edges(k2)
processed_paths = process_paths(paths, ret)
the output processed_paths looks like this:
[['a', frozenset({a0, b0}), 'b'], ['a', frozenset({a1, d2}), 'd', frozenset({d0, e1}), 'e', frozenset({e3, b2}) 'b'], 
 ['a', frozenset({a2, c3}), 'c', frozenset({c1, e0}), 'e', frozenset({e2, d3}), 'd', frozenset({d1, c0}), 'c', frozenset({c2, b1}) 'b']]

Simulation here:
I need a more complex simulation that includes mimics molecular dynamics.
I have these circles that are placed such that they are touching in 1 point
As stated above these points present point through which a spline is drawn through
 but unfortunately the points are too close

I would like to make a simulation that will space the points apart but still not
 make any crossings that were not there before so in essence a connected chain 
 of points that gets simulated so it finds some balanced position where it 
 spaces itself through available space but doesnt fly-off / is still joined together
 with the right connections

One idea is to implement electrostatic repulsive force, and then an attractive force
 for the 2 nearest neighbours of each point in the sequence

Make the code modular, make it fit the outline in this file,
I will also provide you with the code for my simple diffusion simulation that I have right now


#draw_from_layout(circlepack_layout):
plot_splines(sequence, circles) -> image
"""


# More complex simulation __________________________________________

# simulation_parameters.py

# Force coefficients
REPULSIVE_COEFF = 10.0  # Strength of repulsive force
ATTRACTIVE_COEFF = 0.5  # Strength of attractive force
ATTRACTIVE_COEFF = 50.5  # Strength of attractive force

# Simulation parameters
TIME_STEP = 0.001  # Time step for the simulation
DAMPING = 0.9     # Damping factor to stabilize the simulation
MAX_DISPLACEMENT = 0.1  # Maximum displacement per step
MAX_DISPLACEMENT = 0.5  # Maximum displacement per step

# simulation.py

def prep_sim(ret):
    """
    Prepare simulation by initializing velocities and forces.

    Parameters:
    - ret: dict of circle data

    Returns:
    - circles: dict of SimCircle objects
    """
    circles = {}
    for key, circle in ret.items():
        circles[key] = SimCircle(center=circle.center, radius=circle.radius)
        circles[key].velocity = 0 + 0j  # Initialize velocity
        circles[key].force = 0 + 0j     # Initialize force
    return circles


def compute_repulsive_forces(circles, repulsive_coeff=REPULSIVE_COEFF):
    """
    Compute repulsive forces between all pairs of circles.

    Parameters:
    - circles: dict of SimCircle objects
    - repulsive_coeff: coefficient for repulsive force
    """
    for circle in circles.values():
        circle.force = 0 + 0j  # Reset force

    circle_list = list(circles.values())
    n = len(circle_list)
    for i in range(n):
        for j in range(i + 1, n):
            ci = circle_list[i]
            cj = circle_list[j]
            delta = ci.center - cj.center
            distance = abs(delta)
            min_distance = ci.radius + cj.radius

            if distance < min_distance and distance > 1e-5:
                # Overlapping circles, apply strong repulsion
                force_magnitude = repulsive_coeff * (min_distance - distance) / distance
                force = force_magnitude * (delta / distance)
                ci.force += force
                cj.force -= force
            elif distance >= min_distance:
                # Optional: apply weaker repulsion to maintain spacing
                force_magnitude = repulsive_coeff / (distance ** 2)
                force = force_magnitude * (delta / distance)
                ci.force += force
                cj.force -= force


def compute_attractive_forces(circles, connections, attractive_coeff=ATTRACTIVE_COEFF):
    """
    Compute attractive forces between connected circles.

    Parameters:
    - circles: dict of SimCircle objects
    - connections: list of tuples representing connected pairs
    - attractive_coeff: coefficient for attractive force
    """
    for (key1, key2) in connections:
        ci = circles.get(key1)
        cj = circles.get(key2)
        if ci and cj:
            delta = cj.center - ci.center
            distance = abs(delta)
            desired_distance = ci.radius + cj.radius  # Desired spacing

            if distance > 0:
                force_magnitude = attractive_coeff * (distance - desired_distance) / distance
                force = force_magnitude * (delta / distance)
                ci.force += force
                cj.force -= force

"""
def process_path_extra_points(sequences, ret):
    """
    # For adding the additional points to the sequence if necessary, 
    #  move function to sequence1.py, and remove sim and spline funcs from there
    #  import them from proper modules
"""
    def pomozna(sequence, ret):

        new_seq = []
        for key, values in ret.items():
            if isinstance(key, set()):
                node_from, node_to = key


        return new_seq

    sequences2 = [pomozna(sequence, ret) for sequence in sequences]

    return sequences2
"""

def extract_connections(ret):
    """
    Extract connected pairs from the ret dictionary.

    Parameters:
    - ret: dict with keys as node labels and frozensets or tuples representing connections

    Returns:
    - connections: list of tuples (node1, node2)
    """
    connections = []
    for key in ret:
        if isinstance(key, frozenset) or isinstance(key, tuple):
            nodes = list(key)
            # For tuples with more than two nodes, connect them pairwise
            if len(nodes) == 2:
                connections.append((nodes[0], nodes[1]))
            elif len(nodes) > 2:
                # Assuming consecutive pairs are connected
                for i in range(len(nodes) - 1):
                    connections.append((nodes[i], nodes[i + 1]))
    return connections


def limit_displacement(displacement, max_displacement=MAX_DISPLACEMENT):
    """
    Limit the displacement to prevent large jumps.

    Parameters:
    - displacement: complex number representing displacement
    - max_displacement: maximum allowed displacement magnitude

    Returns:
    - limited_displacement: complex number within allowed limits
    """
    magnitude = abs(displacement)
    if magnitude > max_displacement:
        return displacement / magnitude * max_displacement
    return displacement


def make_complex_step(circles, connections, repulsive_coeff=REPULSIVE_COEFF, 
                     attractive_coeff=ATTRACTIVE_COEFF, dt=TIME_STEP, 
                     damping=DAMPING, temperature=0.01):
    """
    Perform a simulation step with both repulsive and attractive forces.

    Parameters:
    - circles: dict of SimCircle objects
    - connections: list of tuples representing connected pairs
    - repulsive_coeff: coefficient for repulsive force
    - attractive_coeff: coefficient for attractive force
    - dt: time step
    - damping: damping factor for velocities
    - temperature: current temperature for displacement limiting

    Returns:
    - updated temperature
    """
    # Compute forces
    compute_repulsive_forces(circles, repulsive_coeff)
    compute_attractive_forces(circles, connections, attractive_coeff)

    # Update velocities and positions
    for circle in circles.values():
        # Update velocity with damping
        circle.velocity = (circle.velocity + circle.force * dt) * damping
        displacement = circle.velocity * dt

        # Limit displacement based on temperature
        displacement = limit_displacement(displacement, MAX_DISPLACEMENT * temperature)

        # Update position
        circle.center += displacement

        # Store positions for plotting every 10 iterations
        if len(circle.positions_ot) % 10 == 0:
            circle.positions_ot.append(circle.center)
            circle.radius_ot.append(circle.radius)

    # Cool down the system
    temperature *= 0.95  # Decrease temperature over time
    return temperature


def run_complex_sim(k, ret, max_iterations=1000):
    """
    Run the complex simulation with repulsive and attractive forces.

    Parameters
    ----------
    k : PlanarDiagram object
    ret : dict
        A dictionary of circlepack layout, where the keys are the node labels and
        the values are the SimCircle objects.
    max_iterations : int
        Maximum number of simulation steps.

    Returns
    -------
    circles : dict
        A dictionary of SimCircle objects with updated positions.
    """
    circles = prep_sim(ret)
    connections = extract_connections(ret)

    # Simulation parameters
    koeff = REPULSIVE_COEFF
    dt = TIME_STEP
    damping = DAMPING

    # Simulation loop
    for iteration in tqdm(range(max_iterations), desc="Running Simulation"):
        make_complex_step(circles, connections, koeff, ATTRACTIVE_COEFF, dt, damping)

    return circles


def plot_circles(circles, interval=50):
    """
    Plot the positions of circles at a given interval.

    Parameters:
    - circles: dict of SimCircle objects
    - interval: iteration interval for plotting
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    for key, circle in circles.items():
        circ = plt.Circle((circle.center.real, circle.center.imag), circle.radius, fill=False, edgecolor='blue', alpha=0.5)
        ax.add_artist(circ)
        plt.plot(circle.center.real, circle.center.imag, 'ro')  # Center point

    plt.axis('equal')
    plt.grid(True)
    plt.title('Enhanced Simulation: Circles After MD-style Forces')
    plt.show()

import matplotlib.animation as animation

def animate_simulation(circles_history):
    """
    Animate the simulation steps.

    Parameters:
    - circles_history: list of dicts containing SimCircle objects at each step
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(True)

    patches = []
    for circle in circles_history[0].values():
        patch = plt.Circle((circle.center.real, circle.center.imag), circle.radius, fill=False, edgecolor='blue', alpha=0.5)
        ax.add_patch(patch)
        patches.append(patch)

    def init():
        for patch in patches:
            patch.center = (0, 0)
        return patches

    def animate(i):
        for patch, circle in zip(patches, circles_history[i].values()):
            patch.center = (circle.center.real, circle.center.imag)
        return patches

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(circles_history), interval=50, blit=True)
    plt.show()



# simulation.py

if __name__ == "__main__":
    # Define your knot using PD notation
    s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
    # Large:
    s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
    #s = "V[0,1,2], V[0,2,1]"
    #s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
    # Problematic:
    s = "X[0,1,2,3],X[4,5,6,7],X[1,8,9,10],X[11,12,13,9],X[14,15,7,16],X[17,18,19,13],X[10,20,17,12],X[20,19,15,14],V[3,21,22],V[5,23,24],V[6,24,16],V[4,18,23],V[0,22,8],V[2,11,21]"
    k2 = from_pd_notation(s)
    print(k2)

    ret = circlepack_layout(k2)

    print("\nInitial Circle Positions:")
    filtered_ret = extract_main_points_and_connections(ret)
    # ret1 = run_sim(filtered_ret)
    ret1 = {key: ret[key].center for key in ret}
    pp.pprint(ret1)

    # Plot initial Bezier curves
    draw_from_layout(k2, ret, with_labels=False, with_title=True)

    # Plot initial circles
    ret0 = prep_sim(filtered_ret)
    #plot_circles(ret0, 0)
    print("\nInitial Simulation")
    #ret1 = run_complex_sim(k2, filtered_ret)
    #ret2 = run_complex_sim(k2, filtered_ret, max_iterations=10000)
    #ret3 = run_complex_sim(k2, ret0, max_iterations=1000)
    #ret3 = extract_main_points_and_connections(ret3)
    #plot_circles(prep_sim(ret1), 50)
    #ret2 = {key: ret1[key].center for key in ret1}
    #pp.pprint(ret2)

    ret_added_points = add_additional_points(filtered_ret)
    ret_added_sim = run_complex_sim(k2, ret_added_points)
    ret1 = remove_additional_points(ret_added_sim)

    # Plot initial spline curves
    print("\nPaths:")
    paths = edges(k2)
    print(paths)
    processed_paths = process_paths(paths, ret)
    print(processed_paths)
    circles_data0 = build_all_circles_data(processed_paths, ret0)
    circles_data1 = build_all_circles_data(processed_paths, ret1)
    #circles_data2 = build_all_circles_data(processed_paths, ret2)
    #circles_data3 = build_all_circles_data(processed_paths, ret3)
    print("\nSplines Circles data")
    #pp.pprint(circles_data0)
    skip_indices = build_skip_indices(sequences=processed_paths, skip_radius=1)
    skip_indices_for_all_paths = build_skip_indices_for_all_paths(processed_paths)


    plot_splines(circles_data0, skip_indices_for_all_paths)
    plot_splines(circles_data1, skip_indices_for_all_paths)
    #plot_splines(circles_data0, None)
    #plot_splines(circles_data1, None)
    #plot_splines(circles_data2, 301)
    #plot_splines(circles_data3, 302)
    plt.show()

    exit()

    # Run the enhanced complex simulation
    print("\nRunning Enhanced Simulation")
    enhanced_circles = run_complex_sim(k2, filtered_ret, max_iterations=1000)

    # Plot circles after enhanced simulation
    plot_circles(enhanced_circles, 50)

    # Extract positions for spline plotting
    enhanced_ret2 = {key: enhanced_circles[key].center for key in enhanced_circles}
    pp.pprint(enhanced_ret2)

    # Plot enhanced spline curves
    enhanced_circles_data = build_all_circles_data(processed_paths, enhanced_circles)
    plot_splines(enhanced_circles_data)

    # Optional: Animate the simulation
    # Store history if you modify run_complex_sim to keep snapshots
    # animate_simulation(circles_history)

exit()


