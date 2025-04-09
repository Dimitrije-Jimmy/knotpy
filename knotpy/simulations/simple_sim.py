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
__version__ = '0.1'
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


s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
#s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
#s = "V[0,1,2], V[0,2,1]"
#s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
#s = "V[2,6,1], V[7,1,8], V[3,8,4], V[5,4,6], X[5,2,7,3]"
print("\n")
print(s)
k2 = from_pd_notation(s)
print(k2)

# always named ret
ret = circlepack_layout(k2)

print("\n")
pp.pprint(ret)

class SimCircle(Circle):
    def __init__(self, center, radius):
        super().__init__(center, radius)
        self.force = 0+0j      # Initialize force as a complex number
        self.velocity = 0+0j   # Initialize velocity as a complex number
        self.mass = 1.0        # Assume unit mass for simplicity


circlepack_layout_sim = {}
for key, circle in ret.items():
    sim_circle = SimCircle(circle.center, circle.radius)
    circlepack_layout_sim[key] = sim_circle

# Extract circles and create mapping
# List of all SimCircle objects
circles = list(circlepack_layout_sim.values())

# Initialize tangency constraints
tangency_constraints = []

for key, circle in circlepack_layout_sim.items():
    if isinstance(key, frozenset) or isinstance(key, tuple):
        # The circle should be tangent to the circles corresponding to the elements in the key
        for element in key:
            # Map 'a0', 'a1', etc., to 'a'; 'b0', 'b1', etc., to 'b'; 'c0', 'c1', etc., to 'c'
            base_label = element[0]  # 'a', 'b', or 'c'
            connected_circle = circlepack_layout_sim.get(base_label)
            if connected_circle is not None:
                tangency_constraints.append((circle, connected_circle))
    else:
        # For crossing circles ('a', 'b', 'c'), we may have additional constraints
        pass  # Assuming crossing circles don't have additional constraints here


# Simulation parameters
k_rep = 1.0    # Repulsion constant
k_att = 5.0    # Attraction constant
damping = 0.1  # Damping coefficient
dt = 0.01      # Time step
max_iterations = 100  # Maximum number of iterations

# Function to compute repulsive forces
def compute_repulsive_forces(circles):
    for i, ci in enumerate(circles):
        for cj in circles[i+1:]:
            delta = cj.center - ci.center
            distance = abs(delta)
            min_distance = ci.radius + cj.radius
            overlap = min_distance - distance
            if overlap > 0:
                # Compute repulsive force
                force_magnitude = k_rep * overlap
                force_direction = delta / distance if distance != 0 else 1+0j
                force = force_magnitude * force_direction
                ci.force -= force
                cj.force += force

def compute_attractive_forces(constraints):
    for c1, c2 in constraints:
        delta = c2.center - c1.center
        distance = abs(delta)
        target_distance = c1.radius + c2.radius
        gap = distance - target_distance
        # Compute attractive force
        force_magnitude = -k_att * gap
        force_direction = delta / distance if distance != 0 else 1+0j
        force = force_magnitude * force_direction
        c1.force -= force
        c2.force += force


# Simulation loop
positions_over_time = []

for iteration in range(max_iterations):
    # Reset forces
    for circle in circles:
        circle.force = 0+0j

    # Compute forces
    compute_repulsive_forces(circles)
    #compute_attractive_forces(tangency_constraints)

    # Update velocities and positions
    for circle in circles:
        # Update velocity
        circle.velocity += (circle.force / circle.mass) * dt
        # Apply damping
        circle.velocity *= (1 - damping * dt)
        # Update position
        circle.center += circle.velocity * dt

    # Store positions for plotting every 10 iterations
    if iteration % 10 == 0:
        positions_over_time.append([(circle.center, circle.radius) for circle in circles])

# Visualization function
def plot_circles(circles_data, iteration):
    fig, ax = plt.subplots(figsize=(6,6))
    for center, radius in circles_data:
        circle = plt.Circle((center.real, center.imag), radius, fill=False, edgecolor='blue')
        ax.add_artist(circle)
        # Optionally, plot the center
        ax.plot(center.real, center.imag, 'ro', markersize=2)
    ax.set_aspect('equal', 'box')
    ax.set_title(f'Iteration {iteration}')
    # Adjust plot limits based on circle positions
    all_centers = np.array([center for center, _ in circles_data])
    min_x, max_x = np.min(all_centers.real) - 1, np.max(all_centers.real) + 1
    min_y, max_y = np.min(all_centers.imag) - 1, np.max(all_centers.imag) + 1
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.grid(True)
    plt.show()
"""
# Plotting the positions
for idx, circles_data in enumerate(positions_over_time):
    iteration = idx * 10
    plot_circles(circles_data, iteration)





exit()
"""
draw_from_layout(k2, ret, with_labels=True, with_title=True)
plt.show()

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
    print("\n new dict")
    import itertools
    ret = dict(itertools.islice(ret.items(), 0, None, 10))
    pp.pprint(ret)


    draw_from_layout(k2, ret, with_labels=True, with_title=True)
    
    # Example usage:
    circle_positions = extract_circle_positions(ret)
    for circle, position in circle_positions.items():
        print(f"Circle {circle}: Position {position}")

    draw_from_layout(k2, ret, with_labels=True, with_title=True)
    plt.show()