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

import math
import cmath
import random

def initialize_circles_and_connections(ret):
    circles = {}  # Key: identifier (could be any hashable type), Value: Circle object
    connections = []  # List of tuples representing connections between circle identifiers

    for key, circle in ret.items():
        circles[key] = circle  # Include all circles

        # Determine connections based on key type
        if isinstance(key, frozenset):
            keys = list(key)
            point_ids = [str(k)[0] for k in keys]
            #print(point_ids)
            if len(point_ids) == 2:
                connections.append(tuple(point_ids))
            elif len(point_ids) > 2:
                # Handle connections involving more than two points
                # For simplicity, create pairwise connections
                for i in range(len(point_ids)):
                    for j in range(i + 1, len(point_ids)):
                        connections.append((point_ids[i], point_ids[j]))
        # If necessary, handle other key types (e.g., tuples)
        elif isinstance(key, tuple):
            point_ids = [str(k)[0] for k in key]
            print(point_ids)
            if len(point_ids) >= 2:
                for i in range(len(point_ids)):
                    for j in range(i + 1, len(point_ids)):
                        connections.append((point_ids[i], point_ids[j]))
        # Strings or other types are considered individual circles without explicit connections

    return circles, connections

def compute_energy(circles, connections):
    energy = 0.0

    # Penalty for overlapping circles
    circle_list = list(circles.values())
    for i in range(len(circle_list)):
        for j in range(i + 1, len(circle_list)):
            c1 = circle_list[i]
            c2 = circle_list[j]
            dist = abs(c1.center - c2.center)
            min_dist = c1.radius + c2.radius
            if dist < min_dist:
                # Overlapping
                overlap = min_dist - dist
                energy += overlap ** 2  # Quadratic penalty

    # Reward for even distribution (optional)
    # For simplicity, we can minimize the variance of distances between circles

    return energy


def compute_forces(circles, connections):
    forces = {key: 0 + 0j for key in circles.keys()}  # Initialize forces to zero

    # Repulsive forces between all circles to prevent overlap
    circle_keys = list(circles.keys())
    for i in range(len(circle_keys)):
        for j in range(i + 1, len(circle_keys)):
            key_i = circle_keys[i]
            key_j = circle_keys[j]
            c1 = circles[key_i]
            c2 = circles[key_j]
            dist_vector = c1.center - c2.center
            dist = abs(dist_vector)
            min_dist = c1.radius + c2.radius

            if dist < min_dist:
                # Overlapping or touching; apply repulsive force
                overlap = min_dist - dist
                if dist != 0:
                    direction = dist_vector / dist
                else:
                    angle = random.uniform(0, 2 * math.pi)
                    direction = cmath.rect(1, angle)
                force_magnitude = overlap
                force = direction * force_magnitude
                forces[key_i] += force
                forces[key_j] -= force

    # Attractive forces to maintain connections
    for key_i, key_j in connections:
        c1 = circles[key_i]
        c2 = circles[key_j]
        dist_vector = c1.center - c2.center
        dist = abs(dist_vector)
        desired_dist = c1.radius + c2.radius

        if dist != desired_dist:
            delta = dist - desired_dist
            if dist != 0:
                direction = dist_vector / dist
            else:
                angle = random.uniform(0, 2 * math.pi)
                direction = cmath.rect(1, angle)
            force_magnitude = delta
            force = direction * force_magnitude
            forces[key_i] -= force
            forces[key_j] += force

    return forces

def update_positions(circles, forces, dt):
    for key, force in forces.items():
        c = circles[key]
        c.center += force * dt

def adjust_radii(circles, max_radius=None):
    # Compute average radius
    radii = [c.radius for c in circles.values()]
    avg_radius = sum(radii) / len(radii)

    # Optionally set a maximum radius
    target_radius = max_radius if max_radius else avg_radius

    for c in circles.values():
        # Increase radius towards target_radius if it doesn't cause overlaps
        c.radius = min(c.radius + 0.1, target_radius)  # Adjust increment as needed


def check_constraints(circles, connections):
    # Check tangency between connected circles
    for key_i, key_j in connections:
        c1 = circles[key_i]
        c2 = circles[key_j]
        dist = abs(c1.center - c2.center)
        desired_dist = c1.radius + c2.radius
        if not math.isclose(dist, desired_dist, rel_tol=1e-4):
            print(f"Constraint violation between {key_i} and {key_j}")
            return False

    # Check for overlaps between non-connected circles
    circle_keys = list(circles.keys())
    for i in range(len(circle_keys)):
        for j in range(i + 1, len(circle_keys)):
            key_i = circle_keys[i]
            key_j = circle_keys[j]
            if (key_i, key_j) in connections or (key_j, key_i) in connections:
                continue
            c1 = circles[key_i]
            c2 = circles[key_j]
            print(c1, c2)
            dist = abs(c1.center - c2.center)
            min_dist = c1.radius + c2.radius
            if dist < min_dist - 1e-4:
                print(f"Overlap detected between {key_i} and {key_j}")
                return False

    return True


def simulate(ret, iterations=1000, dt=0.01):
    # Initialize circles and connections
    circles, connections = initialize_circles_and_connections(ret)

    for iteration in range(iterations):
        forces = compute_forces(circles, connections)
        update_positions(circles, forces, dt)
        adjust_radii(circles)

        if iteration % 100 == 0:
            energy = compute_energy(circles, connections)
            print(f"Iteration {iteration}, Energy: {energy}")

    if check_constraints(circles, connections):
        print("Simulation completed successfully with all constraints satisfied.")
    else:
        print("Simulation completed with constraint violations.")

    # Update the original ret dictionary with new Circle objects
    for key in ret.keys():
        ret[key] = circles[key]

    return ret


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
#draw_from_layout(k2, filtered_ret, with_labels=True, with_title=True)

print("\n")
# Run the main function
final_ret = simulate(ret)
#final_ret = simulate(filtered_ret)


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
#draw_from_layout(k2, filtered_ret, with_labels=True, with_title=True)
#draw_from_layout(k2, final_ret, with_labels=True, with_title=True)


# Before simulation
plot_ret(ret, title="Before Simulation")
# After simulation
plot_ret(final_ret, title="After Simulation")




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