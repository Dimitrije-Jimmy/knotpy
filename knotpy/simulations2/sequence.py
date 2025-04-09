"""
Sequencing functions
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
from simulation import prep_sim, compute_repulsive_forces, limit_displacement, make_step, run_sim, plot_circles

from spline import plot_spline_normal, build_circles_data, extract_points
from spline import plot_bezier_curve_matplotlib,bezier_curve_matplotlib, bezier_curve_matplotlib_multiple, get_positions

__all__ = ['process_paths', 'process_path', 'plot_splines_sequence', 'build_all_circles_data']
__version__ = 'god knows'


# Start of multiple Seuqencing Splines __________________________________________________

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

def plot_splines_sequence(all_circles_data, num_spline_points=300):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    from scipy.interpolate import CubicSpline

    fig, ax = plt.subplots()

    all_centers = []

    # Define colors for each spline
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    color_cycle = iter(colors)

    bc_type = 'periodic' if len(all_circles_data) == 1 else 'not-a-knot'

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
        t_new = np.linspace(t[0], t[-1], num_spline_points)

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
        x_spline = make_interp_spline(t, x_points, k=k, bc_type=bc_type)
        y_spline = make_interp_spline(t, y_points, k=k, bc_type=bc_type)
        #x_spline = CubicSpline(t, x_points, bc_type='periodic')
        #y_spline = CubicSpline(t, y_points, bc_type='periodic')

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
    plt.title(f'Splines Through Knot Diagram Points - {num_spline_points}')
    plt.legend()
    #plt.show()



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
    plot_splines_sequence(all_circles_data)
    plot_splines_sequence(all_circles_data2)

    plt.show()


