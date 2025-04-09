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
__version__ = '0.3'
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


# NEW SIM _____________________________________________________________________
import random

def prep_func(ret):
    circles = {}  # Key: circle identifier, Value: Circle object
    connections = []  # List of tuples (circle_id1, circle_id2)

    for key, circle in ret.items():
        # All circles
        circles[key] = circle
        # Connections
        if isinstance(key, frozenset):
            ids = [str(k)[0] for k in key]
            #print(ids)
            if len(ids) == 2:
                connections.append((ids[0], ids[1]))
            elif len(ids) > 2:
                # Handle multi-point connections
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        connections.append((ids[i], ids[j]))

    return circles, connections

def force_directed_layout(circles, connections, iterations=1000, dt=0.01):
    k_rep = 1.0
    k_att = 1.0

    positions = {key: circle.center for key, circle in circles.items()}
    radii = {key: circle.radius for key, circle in circles.items()}
    mass = 1.0  # Assuming unit mass for all circles
    damping = 0.9  # To reduce oscillations
    
    # Initialize velocities
    velocities = {key: complex(0, 0) for key in circles.keys()}
    
    for _ in range(iterations):
        forces = {key: complex(0, 0) for key in circles.keys()}
        
        # Repulsive forces between all pairs of circles
        circle_keys = list(circles.keys())
        for i in range(len(circle_keys)):
            for j in range(i + 1, len(circle_keys)):
                key_i = circle_keys[i]
                key_j = circle_keys[j]
                pos_i = positions[key_i]
                pos_j = positions[key_j]
                delta = pos_i - pos_j
                distance = abs(delta)
                min_distance = radii[key_i] + radii[key_j]
                
                if distance < min_distance:
                    # Overlapping or touching; apply repulsive force
                    if distance == 0:
                        # Avoid division by zero
                        direction = complex(random.uniform(-1, 1), random.uniform(-1, 1))
                        distance = abs(direction)
                    else:
                        direction = delta / distance
                    # Repulsive force magnitude
                    overlap = min_distance - distance
                    force_magnitude = overlap * k_rep
                    force = direction * force_magnitude
                    forces[key_i] += force
                    forces[key_j] -= force
        
        # Attractive forces between connected circles
        for key_i, key_j in connections:
            pos_i = positions[key_i]
            pos_j = positions[key_j]
            delta = pos_i - pos_j
            distance = abs(delta)
            desired_distance = radii[key_i] + radii[key_j]
            if distance != desired_distance:
                if distance == 0:
                    # Avoid division by zero
                    direction = complex(random.uniform(-1, 1), random.uniform(-1, 1))
                    distance = abs(direction)
                else:
                    direction = delta / distance
                # Attractive force magnitude
                displacement = distance - desired_distance
                force_magnitude = displacement * k_att
                force = -direction * force_magnitude
                forces[key_i] += force
                forces[key_j] -= force
        
        # Update velocities and positions
        for key in circles.keys():
            # F = m * a, but assuming m = 1
            acceleration = forces[key] / mass
            velocities[key] = (velocities[key] + acceleration * dt) * damping
            positions[key] += velocities[key] * dt
        
        # Optionally, apply constraints or corrections
        # e.g., prevent positions from drifting too far
    
    # Update circle centers
    for key in circles.keys():
        circles[key].center = positions[key]

    return circles


# OLD STUFF ___________________________________________________________________

# Visualization function
def plot_circles(circles_data, iteration):
    fig, ax = plt.subplots(figsize=(6,6))
    for center, radius in circles_data:
        circle = plt.Circle((center.real, center.imag), radius, fill=False, edgecolor='blue')
        ax.add_artist(circle)
        # Plot the center
        ax.plot(center.real, center.imag, 'ro', markersize=2)
    ax.set_aspect('equal', 'box')
    ax.set_title(f'Iteration {iteration}')
    # Adjust plot limits
    all_centers = np.array([center for center, _ in circles_data])
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

#s = "('PlanarDiagram', {'name': '+t3_1#-3_1(1).2'}, [('Vertex', 'a', (('IngoingEndpoint', 'c', 2, {}), ('OutgoingEndpoint', 'b', 1, {'color': 1}), ('OutgoingEndpoint', 'd', 0, {})), {}), ('Vertex', 'b', (('IngoingEndpoint', 'd', 1, {}), ('IngoingEndpoint', 'a', 1, {'color': 1}), ('OutgoingEndpoint', 'g', 0, {})), {}), ('Crossing', 'c', (('IngoingEndpoint', 'f', 3, {}), ('IngoingEndpoint', 'f', 2, {}), ('OutgoingEndpoint', 'a', 0, {}), ('OutgoingEndpoint', 'e', 0, {})), {}), ('Crossing', 'd', (('IngoingEndpoint', 'a', 2, {}), ('OutgoingEndpoint', 'b', 0, {}), ('OutgoingEndpoint', 'g', 3, {}), ('IngoingEndpoint', 'h', 2, {})), {}), ('Crossing', 'e', (('IngoingEndpoint', 'c', 3, {}), ('IngoingEndpoint', 'h', 1, {}), ('OutgoingEndpoint', 'f', 1, {}), ('OutgoingEndpoint', 'f', 0, {})), {}), ('Crossing', 'f', (('IngoingEndpoint', 'e', 3, {}), ('IngoingEndpoint', 'e', 2, {}), ('OutgoingEndpoint', 'c', 1, {}), ('OutgoingEndpoint', 'c', 0, {})), {}), ('Crossing', 'g', (('IngoingEndpoint', 'b', 2, {}), ('OutgoingEndpoint', 'h', 0, {}), ('OutgoingEndpoint', 'h', 3, {}), ('IngoingEndpoint', 'd', 2, {})), {}), ('Crossing', 'h', (('IngoingEndpoint', 'g', 1, {}), ('OutgoingEndpoint', 'e', 1, {}), ('OutgoingEndpoint', 'd', 3, {}), ('IngoingEndpoint', 'g', 2, {})), {})])"
#k = from_knotpy_notation(s)
s = "V[0,1,2],V[3,1,4],X[5,6,0,7],X[2,3,8,9],X[7,10,11,12],X[12,11,6,5],X[4,13,14,8],X[13,10,9,14]"


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
draw_from_layout(k2, ret, with_labels=True, with_title=True)
plt.show()
draw_from_layout(k2, filtered_ret, with_labels=True, with_title=True)
plt.show()

print("\n")
# Run the main function

circles, connections = prep_func(filtered_ret)

print("\n")
pp.pprint(circles)
print("\n")
pp.pprint(connections)

final_ret = force_directed_layout(circles, connections, 1000, 0.01)


# Before simulation
print("Before simulation:")
for key, circle in ret.items():
    print(f"Key: {key}, Circle: {circle}")
# After simulation
print("\nAfter simulation:")
for key, circle in final_ret.items():
    print(f"Key: {key}, Circle: {circle}")


def plot_ret(ret, title="Circles"):
    fig, ax = plt.subplots()
    for key, c in ret.items():
        circle_patch = plt.Circle((c.center.real, c.center.imag), c.radius, fill=False)
        ax.add_patch(circle_patch)
        ax.plot(c.center.real, c.center.imag, 'o')  # Circle center

    ax.set_aspect('equal', adjustable='datalim')
    plt.title(title)
    plt.show()


print("\n")
pp.pprint(final_ret)


# Before simulation
plot_ret(ret, title="Before Simulation")
plot_ret(filtered_ret, title="Before Simulation")
# After simulation
plot_ret(final_ret, title="After Simulation")

draw_from_layout(k2, filtered_ret, with_labels=True, with_title=True)
#draw_from_layout(k2, final_ret, with_labels=True, with_title=True)



exit()
#pp.pprint(positions_over_time[0])
#pp.pprint(positions_over_time[0])

# Plotting the positions
#for idx, circles_data in enumerate(positions_over_time):
#    iteration = idx * 10
#    plot_circles(circles_data, iteration)

#print(len(positions_over_time))
#pp.pprint(positions_over_time[0])
plot_circles(positions_over_time[0], 0)
plot_circles(positions_over_time[9], 100)


#pp.pprint(list(circlepack_layout_sim.values()) == positions_over_time[0])

#draw_from_layout(k2, circlepack_layout_sim, with_labels=True, with_title=True)


#draw_from_layout(k2, ret, with_labels=True, with_title=True)
#plt.show()


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