import numpy as np
import matplotlib.pyplot as plt

# Initialize circles with random positions and radii
#np.random.seed(42)  # For reproducibility

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
num_circles = len(labels)

# Random initial positions and radii
positions = np.random.rand(num_circles, 2) * 10  # Positions in 10x10 space
radii = np.random.rand(num_circles) * 5 + 1      # Radii between 1 and 3

circles = {label: {'position': positions[i], 'radius': radii[i]} for i, label in enumerate(labels)}

# Define connections between circles
connections = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('f', 'g')]

def simulation(circles, connections):
    # Simulation parameters
    num_iterations = 1000
    learning_rate = 0.01
    average_radius = np.mean(radii)
    radius_adjust_rate = 0.005

    # Simulation loop
    for iteration in range(num_iterations):
        # Initialize forces and radius adjustments
        forces = {label: np.array([0.0, 0.0]) for label in labels}
        radius_changes = {label: 0.0 for label in labels}
        
        # Apply forces based on connections
        for (label1, label2) in connections:
            circle1 = circles[label1]
            circle2 = circles[label2]
            pos1 = circle1['position']
            pos2 = circle2['position']
            r1 = circle1['radius']
            r2 = circle2['radius']
            
            # Desired distance is touching (sum of radii)
            desired_distance = r1 + r2
            delta = pos2 - pos1
            distance = np.linalg.norm(delta)
            if distance == 0:
                # Prevent division by zero
                direction = np.random.rand(2) - 0.5
            else:
                direction = delta / distance
            
            # Spring force proportional to distance error
            force_magnitude = (distance - desired_distance)
            force = direction * force_magnitude * learning_rate
            forces[label1] += force
            forces[label2] -= force  # Equal and opposite force
        
        # Apply repulsion between non-connected circles
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels):
                if i >= j:
                    continue  # Avoid double computation
                if (label1, label2) in connections or (label2, label1) in connections:
                    continue  # Already handled connected circles
                circle1 = circles[label1]
                circle2 = circles[label2]
                pos1 = circle1['position']
                pos2 = circle2['position']
                r1 = circle1['radius']
                r2 = circle2['radius']
                
                delta = pos2 - pos1
                distance = np.linalg.norm(delta)
                if distance == 0:
                    # Prevent division by zero
                    direction = np.random.rand(2) - 0.5
                    distance = np.linalg.norm(direction)
                    direction /= distance
                else:
                    direction = delta / distance
                
                overlap = r1 + r2 - distance
                if overlap > 0:
                    # Repulsive force to separate overlapping circles
                    force = direction * overlap * learning_rate
                    forces[label1] -= force
                    forces[label2] += force
        
        # Update positions
        for label in labels:
            circles[label]['position'] += forces[label]
        
        # Adjust radii towards average radius
        for label in labels:
            radius_difference = average_radius - circles[label]['radius']
            radius_changes[label] += radius_difference * radius_adjust_rate
            circles[label]['radius'] += radius_changes[label]
        
        # Optionally, enforce minimal and maximal radii to allow variance
        for label in labels:
            circles[label]['radius'] = np.clip(circles[label]['radius'], 0.8 * average_radius, 1.2 * average_radius)

    return circles

def visualise(circles, connections, labels):
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
            if label == label1:
                pos2 = circles[label2]['position']
                plt.plot([pos[0], pos2[0]], [pos[1], pos2[1]], 'r--')

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    plt.title('Final Configuration of Circles')
    plt.show()


visualise(circles, connections, labels)
circles_end = simulation(circles, connections)
visualise(circles_end, connections, labels)