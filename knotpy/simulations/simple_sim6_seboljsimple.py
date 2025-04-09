import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as pltCircle

def compute_forces(circles, connections, k_rep=1.0, k_att=1.0):
    forces = {label: 0 + 0j for label in circles.keys()}

    # Repulsive forces between all pairs of circles
    circle_labels = list(circles.keys())
    for i in range(len(circle_labels)):
        for j in range(i + 1, len(circle_labels)):
            label_i = circle_labels[i]
            label_j = circle_labels[j]
            circle_i = circles[label_i]
            circle_j = circles[label_j]
            delta = circle_i['center'] - circle_j['center']
            distance = abs(delta)
            min_distance = circle_i['radius'] + circle_j['radius']

            if distance < min_distance:
                # Overlapping or touching; apply repulsive force
                if distance == 0:
                    # Avoid division by zero
                    angle = random.uniform(0, 2 * np.pi)
                    direction = np.exp(1j * angle)
                else:
                    direction = delta / distance
                # Repulsive force magnitude
                overlap = min_distance - distance
                force_magnitude = k_rep * overlap
                force = direction * force_magnitude
                forces[label_i] += force
                forces[label_j] -= force

    # Attractive forces between connected circles
    for label_i, label_j in connections:
        circle_i = circles[label_i]
        circle_j = circles[label_j]
        delta = circle_i['center'] - circle_j['center']
        distance = abs(delta)
        desired_distance = circle_i['radius'] + circle_j['radius']
        if distance != desired_distance:
            if distance == 0:
                # Avoid division by zero
                angle = random.uniform(0, 2 * np.pi)
                direction = np.exp(1j * angle)
            else:
                direction = delta / distance
            # Attractive force magnitude
            displacement = distance - desired_distance
            force_magnitude = k_att * displacement
            force = -direction * force_magnitude
            forces[label_i] += force
            forces[label_j] -= force

    return forces

def update_positions(circles, forces, dt=0.01, damping=0.9):
    for label in circles.keys():
        circle = circles[label]
        force = forces[label]
        # Assuming unit mass, acceleration equals force
        acceleration = force
        # Initialize velocity if not present
        if 'velocity' not in circle:
            circle['velocity'] = 0 + 0j
        # Update velocity
        circle['velocity'] = (circle['velocity'] + acceleration * dt) * damping
        # Update position
        circle['center'] += circle['velocity'] * dt

def simulate(circles, connections, iterations=1000, dt=0.01):
    for iteration in range(iterations):
        forces = compute_forces(circles, connections)
        update_positions(circles, forces, dt=dt)

        # Optional: print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}")
            # Optionally plot intermediate states
            # plot_circles(circles, connections, iteration=iteration)

def plot_circles(circles, connections, iteration=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot connections
    for label_i, label_j in connections:
        center_i = circles[label_i]['center']
        center_j = circles[label_j]['center']
        ax.plot([center_i.real, center_j.real], [center_i.imag, center_j.imag], 'k--', zorder=1)

    # Plot circles
    for label, circle in circles.items():
        center = circle['center']
        radius = circle['radius']
        circle_patch = pltCircle((center.real, center.imag), radius, fill=False, color='blue', lw=2, zorder=2)
        ax.add_patch(circle_patch)
        # Label the circles
        ax.text(center.real, center.imag, label, ha='center', va='center', zorder=3)

    ax.set_aspect('equal')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title(f"Force-Directed Layout{' - Iteration ' + str(iteration) if iteration is not None else ''}")
    plt.show()

def main():
    # Define circle labels
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    # Initialize circles with random positions and radii
    circles = {}
    for label in labels:
        # Random initial positions within a range
        center = complex(random.uniform(-10, 10), random.uniform(-10, 10))
        # Random radii between 1 and 3
        radius = random.uniform(1, 3)
        circles[label] = {
            'center': center,
            'radius': radius
        }

    # Define interesting connections between circles
    connections = [
        ('a', 'b'),
        ('b', 'c'),
        ('c', 'd'),
        ('d', 'e'),
        ('e', 'a'),  # Forming a pentagon shape
        ('b', 'd'),  # Diagonal connection
        ('f', 'g')   # Separate connection
    ]

    # Initial plot
    plot_circles(circles, connections, iteration=0)

    # Run the simulation
    simulate(circles, connections, iterations=1000, dt=0.01)

    # Final plot
    plot_circles(circles, connections, iteration='Final')

if __name__ == "__main__":
    main()
