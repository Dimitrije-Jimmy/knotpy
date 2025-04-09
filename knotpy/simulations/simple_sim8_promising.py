import numpy as np
import matplotlib.pyplot as plt

def initialize_circles(labels):
    # Random initial positions and radii
    np.random.seed(42)  # For reproducibility
    num_circles = len(labels)
    positions = np.random.rand(num_circles, 2) * 10  # Positions in 10x10 space
    radii = np.random.rand(num_circles) * 2 + 1      # Radii between 1 and 3
    circles = {label: {'position': positions[i], 'radius': radii[i]} for i, label in enumerate(labels)}
    return circles

def simulation(circles, connections, labels):
    # Simulation parameters
    num_iterations = 100000
    tolerance = 1e-10
    average_radius = np.mean([circles[label]['radius'] for label in labels])
    radius_adjust_rate = 0.01

    # Precompute pairs for efficiency
    connected_pairs = set(connections)
    all_pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i+1, len(labels))]
    non_connected_pairs = [pair for pair in all_pairs if pair not in connected_pairs and (pair[1], pair[0]) not in connected_pairs]

    for iteration in range(num_iterations):
        max_displacement = 0

        # Adjust radii towards average radius
        for label in labels:
            radius_difference = average_radius - circles[label]['radius']
            circles[label]['radius'] += radius_difference * radius_adjust_rate
            # Enforce minimal and maximal radii to allow variance
            circles[label]['radius'] = np.clip(circles[label]['radius'], 0.8 * average_radius, 1.2 * average_radius)

        # Adjust positions to satisfy constraints
        # Connected circles should be exactly touching
        for (label1, label2) in connections:
            circle1 = circles[label1]
            circle2 = circles[label2]
            pos1 = circle1['position']
            pos2 = circle2['position']
            r1 = circle1['radius']
            r2 = circle2['radius']

            desired_distance = r1 + r2
            delta = pos2 - pos1
            distance = np.linalg.norm(delta)
            if distance == 0:
                # Random small displacement to avoid overlap
                displacement = (np.random.rand(2) - 0.5) * 0.1
                pos2 += displacement
                delta = pos2 - pos1
                distance = np.linalg.norm(delta)

            # Compute displacement needed
            correction = (delta / distance) * (distance - desired_distance) * 0.5
            # Update positions
            circles[label1]['position'] += correction
            circles[label2]['position'] -= correction
            max_displacement = max(max_displacement, np.linalg.norm(correction))

        # Non-connected circles should not overlap or touch
        for (label1, label2) in non_connected_pairs:
            circle1 = circles[label1]
            circle2 = circles[label2]
            pos1 = circle1['position']
            pos2 = circle2['position']
            r1 = circle1['radius']
            r2 = circle2['radius']

            min_distance = r1 + r2
            delta = pos2 - pos1
            distance = np.linalg.norm(delta)
            if distance == 0:
                # Random small displacement to avoid overlap
                displacement = (np.random.rand(2) - 0.5) * 0.1
                pos2 += displacement
                delta = pos2 - pos1
                distance = np.linalg.norm(delta)

            if distance < min_distance:
                # Compute displacement needed
                correction = (delta / distance) * (min_distance - distance) * 0.5
                # Update positions
                circles[label1]['position'] -= correction
                circles[label2]['position'] += correction
                max_displacement = max(max_displacement, np.linalg.norm(correction))

        # Check for convergence
        if max_displacement < tolerance:
            print(f"Converged after {iteration} iterations.")
            break

    return circles

def visualize(circles, connections, labels):
    # Visualization
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    for label in labels:
        circle = circles[label]
        pos = circle['position']
        radius = circle['radius']
        circle_patch = plt.Circle(pos, radius, fill=False, edgecolor='b')
        ax.add_patch(circle_patch)
        plt.text(pos[0], pos[1], label, ha='center', va='center')
    # Draw connections
    for (label1, label2) in connections:
        pos1 = circles[label1]['position']
        pos2 = circles[label2]['position']
        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'r--')

    ax.set_xlim(-5, 25)
    ax.set_ylim(-5, 25)
    ax.set_aspect('equal')
    plt.title('Configuration of Circles')
    plt.show()

# Main execution
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
circles = initialize_circles(labels)
connections = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), ('f', 'g')]

# Visualize initial positions
visualize(circles, connections, labels)

# Run simulation
circles_end = simulation(circles, connections, labels)

# Visualize final positions
visualize(circles_end, connections, labels)
