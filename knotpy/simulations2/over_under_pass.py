"""
Splines with overpass and underpass
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
from spline import plot_bezier_curve_matplotlib, bezier_curve_matplotlib, bezier_curve_matplotlib_multiple, get_positions

__all__ = ['plot_splines', 'plot_splines_overunder2', 'build_all_circles_data', 'build_skip_indices', 'build_skip_indices_for_all_paths']
__version__ = 'god knows'


# Splines Overpass / Underpass _________________________________________________ 

def expand_skips(skip_indices, radius, path_length):
    expanded = set()
    for i in skip_indices:
        # For example, skip i-1, i, i+1 if within bounds
        for j in range(max(0, i - radius), min(path_length, i + radius + 1)):
            expanded.add(j)
    return expanded

def build_skip_indices(sequences, skip_radius=1):
    skip_indices_for_all_paths = [set() for _ in sequences]
    seen_letters = set()

    for path_idx, path in enumerate(sequences):
        for item_idx, item in enumerate(path):
            if isinstance(item, str) and len(item) == 1:
                if item not in seen_letters:
                    skip_indices_for_all_paths[path_idx].add(item_idx)
                    seen_letters.add(item)

        # Optionally expand each path's skip set
        #skip_indices_for_all_paths[path_idx] = expand_skips(
        #    skip_indices_for_all_paths[path_idx],
        #    skip_radius,
        #    len(path)
        #)
    
    #print("\n", sequences)
    #print("\n", skip_indices_for_all_paths)

    return skip_indices_for_all_paths

def get_label_and_index(node_str):
    """
    Given a node string like 'c2', returns ('c', 2).
    Assumes the node string always has one letter followed by digits.
    """
    if not node_str:
        return None, None
    label = node_str[0]          # 'c'
    index_part = node_str[1]    # '2', or '10' if e.g. 'a10'
    try:
        num_index = int(index_part)
    except ValueError:
        num_index = None
    return label, num_index

def build_skip_indices_for_path(processed_path):
    """
    Given a processed path, return a set of indices to skip,
    based on the rule:
      "If the frozenset before and the frozenset after
       both contain an even subindex for this letter,
       then skip."
    """
    skip_indices = set()

    # We'll iterate from 1..len(path)-2 so we can look
    # safely at path[i-1] and path[i+1].
    for i in range(1, len(processed_path) - 1):
        item = processed_path[i]
        if not isinstance(item, str):
            continue  # skip frozensets, only do letters

        # The letter label, e.g. 'a', 'b', 'c'...
        letter = item

        # The elements before and after should be frozensets,
        # but let's check just in case.
        prev_item = processed_path[i - 1]
        next_item = processed_path[i + 1]

        if not (isinstance(prev_item, frozenset) and isinstance(next_item, frozenset)):
            continue

        # Look for something that starts with the letter in each frozenset
        # e.g. if letter='c', we look for 'c0', 'c1', 'c2' in that frozenset.
        even_before = False
        even_after = False

        # 1) Find the node that begins with `letter` in prev_item
        #    e.g. if letter='c', and prev_item=frozenset({b1, c0}),
        #    then we find 'c0' => numeric index=0 => even.
        for node_str in prev_item:
            #print(node_str)
            if str(node_str).startswith(letter):
                _, num_idx = get_label_and_index(str(node_str))
                if num_idx is not None and (num_idx % 2 == 0):
                    even_before = True
                    break

        # 2) Find the node that begins with `letter` in next_item
        for node_str in next_item:
            if str(node_str).startswith(letter):
                _, num_idx = get_label_and_index(str(node_str))
                if num_idx is not None and (num_idx % 2 == 0):
                    even_after = True
                    break

        if even_before and even_after:
            skip_indices.add(i)

    # add skip_indices start and end
    #skip_indices.add(0)
    #skip_indices.add(len(processed_path) - 1)
    #next_item = processed_path[i + 1]
    #for node_str in next_item:
    #    if str(node_str).startswith(letter):
    #        _, num_idx = get_label_and_index(str(node_str))
    #        if num_idx is not None and (num_idx % 2 == 0):
    #            skip_indices.add(0)
    #
    #prev_item = processed_path[len(processed_path) - 1]
    #for node_str in prev_item:
    #    if str(node_str).startswith(letter):
    #        _, num_idx = get_label_and_index(str(node_str))
    #        if num_idx is not None and (num_idx % 2 == 0):
    #            skip_indices.add(len(processed_path) - 1)

    # Removed because it fails for Vertex V type of connections

    return skip_indices

def build_skip_indices_for_all_paths(processed_paths):
    """
    Returns a list of sets, one for each path,
    storing the indices in that path to skip.
    """
    all_skips = []
    for path in processed_paths:
        skips_for_path = build_skip_indices_for_path(path)
        all_skips.append(skips_for_path)

    print("\n", processed_paths)
    print("\n", all_skips)

    return all_skips


def build_all_circles_data(sequences, circles):
    """
    Build circle data for multiple sequences.

    Parameters:
    - sequences: A list of sequences, where each sequence is a list of keys.
    - circles: A dictionary where keys are node or edge identifiers
               and values are circle objects.

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


def plot_splines(
    all_circles_data, 
    skip_indices_for_all_paths, 
    num_spline_points=300,
    gap_radius=0.3
):
    """
    Plots each path from all_circles_data as a spline, then *removes*
    points near any 'skip indices' from the final plotted curve to
    create an underpass gap.

    Parameters:
    -----------
    - all_circles_data : list of lists of (center, radius)
    - sequences        : same shape as all_circles_data in terms of length
    - skip_indices_for_all_paths : list of sets; skip_indices_for_all_paths[i]
                                   is the set of circle indices in path i to skip
    - num_spline_points: how many points to use for the final spline
    - gap_radius       : how large a region around skip_t to remove in t_new
                         (units: same as the cumulative distance 't')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    color_cycle = iter(colors)
    
    all_centers = []

    for path_idx, circles_data in enumerate(all_circles_data):
        # Extract centers
        centers = [cd[0] for cd in circles_data]
        radii   = [cd[1] for cd in circles_data]

        # Convert to x,y
        x_points = np.array([c.real for c in centers])
        y_points = np.array([c.imag for c in centers])
        # Keep track of them for setting axis limits
        all_centers.extend(centers)

        # If there's fewer than 2 points, skip spline
        if len(centers) < 2:
            continue

        # Create the parameter array t by cumulative distance
        distances = np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        t = cumulative_distances

        # Build the dense parameter array for evaluating the spline
        t_new = np.linspace(t[0], t[-1], num_spline_points)

        # Spline degree
        num_points = len(centers)
        if num_points < 4:
            k = num_points - 1  # e.g. linear or quadratic
        else:
            k = 3  # cubic

        # Build the full spline (with all points)
        x_spline = make_interp_spline(t, x_points, k=k)
        y_spline = make_interp_spline(t, y_points, k=k)

        # Evaluate the spline with all t_new
        x_smooth = x_spline(t_new)
        y_smooth = y_spline(t_new)

        # ----------------------------------------------------
        #   Now remove the points around skip indices
        # ----------------------------------------------------
        skip_idx_set = skip_indices_for_all_paths[path_idx]  # e.g. {3, 5, ...}

        # Build up a list of skip_t values in the param domain
        # i.e. for each skip index i, skip_t = t[i].
        skip_param_values = []
        for i in skip_idx_set:
            if i < len(t):
                skip_t = t[i]
                skip_param_values.append(skip_t)

        # Create a mask for all t_new points
        # True = keep, False = skip
        keep_mask = np.ones_like(t_new, dtype=bool)

        # For each skip param value, remove all t_new[j]
        # where |t_new[j] - skip_t| < gap_radius
        # (You might want to make gap_radius a fraction of total length, etc.)
        for skip_t in skip_param_values:
            keep_mask &= (np.abs(t_new - skip_t) > gap_radius)

        # The final arrays we want to *plot* are the ones we keep
        t_new_kept = t_new[keep_mask]
        x_kept = x_smooth[keep_mask]
        y_kept = y_smooth[keep_mask]

        # We still want to show circles for all original points (centers).
        # We'll do that below. The only difference is the final plotted line
        # will have a gap.

        # ----------------------------------------------------
        #   Plot the "segmented" line
        # ----------------------------------------------------
        # If we just do ax.plot(x_kept, y_kept), it will connect 
        # across the gap. We want to break it at any big jump 
        # in t_new_kept. We can do that by scanning t_new_kept 
        # for discontinuities and plotting separate segments.
        color = next(color_cycle, 'black')  # fallback color if we run out

        if len(t_new_kept) > 1:
            # We'll decide that a gap happens if the difference
            # between consecutive t_new_kept is > some threshold
            # Typically, we just look for any "holes" in the mask that
            # break continuity. A cheap approach:
            jumps = np.where(np.diff(t_new_kept) > (2 * gap_radius))[0]
            # That gives us the indices where we should break the line.
            
            # We'll create segments from [start to jumps[0]], 
            # [jumps[0]+1 to jumps[1]], etc.
            start_idx = 0
            for j in jumps:
                seg_slice = slice(start_idx, j+1)
                ax.plot(x_kept[seg_slice], y_kept[seg_slice], color=color, linewidth=2)
                start_idx = j + 1

            # Plot the last segment
            seg_slice = slice(start_idx, len(x_kept))
            ax.plot(x_kept[seg_slice], y_kept[seg_slice], color=color, linewidth=2)
        else:
            # If everything was removed or there's only 1 point, skip
            pass

        # ----------------------------------------------------
        #   Plot the circles + centers (all original)
        # ----------------------------------------------------
        for (center, radius) in zip(centers, radii):
            c = plt.Circle((center.real, center.imag), radius, fill=False, edgecolor=color, alpha=0.5)
            ax.add_artist(c)
            #ax.plot(center.real, center.imag, 'o', color=color, markersize=5)

    # Adjust the axes
    ax.set_aspect('equal', 'box')
    if all_centers:
        all_centers_arr = np.array(all_centers)
        min_x, max_x = np.min(all_centers_arr.real), np.max(all_centers_arr.real)
        min_y, max_y = np.min(all_centers_arr.imag), np.max(all_centers_arr.imag)
        padding = 1.0
        plt.xlim(min_x - padding, max_x + padding)
        plt.ylim(min_y - padding, max_y + padding)

    plt.grid(True)
    plt.title(f'Splines With Gaps - {num_spline_points} points')
    # plt.show()


def plot_splines_overunder2(
    all_circles_data, 
    skip_indices_for_all_paths, 
    num_spline_points=300,
    gap_fraction=0.03,   # <--- Gap is now a fraction of the path length
    padding_fraction=0.1 # <--- 10% padding around the bounding box
):
    """
    Plots each path from all_circles_data as a spline, then *removes*
    points near any 'skip indices' from the final plotted curve to
    create an underpass gap.

    Parameters:
    -----------
    - all_circles_data : list of lists of (center, radius)
    - skip_indices_for_all_paths : list of sets; skip_indices_for_all_paths[i]
                                   is the set of circle indices in path i to skip
    - num_spline_points: how many points to use for the final spline
    - gap_fraction     : fraction of the path length used as gap radius
    - padding_fraction : fraction of the bounding box used for axis padding
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline


    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    color_cycle = iter(colors)
    
    all_centers = []

    bc_type = 'periodic' if len(all_circles_data) == 1 else 'not-a-knot'
    

    for path_idx, circles_data in enumerate(all_circles_data):
        # Extract centers and radii
        centers = [cd[0] for cd in circles_data]
        radii   = [cd[1] for cd in circles_data]

        # If fewer than 2 points, skip spline
        if len(centers) < 2:
            continue

        # Convert centers to x,y arrays
        x_points = np.array([c.real for c in centers])
        y_points = np.array([c.imag for c in centers])
        
        # Track for axis-limits
        all_centers.extend(centers)

        # Create parameter t by cumulative distance
        distances = np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        t = cumulative_distances

        # Build dense parameter for spline
        t_new = np.linspace(t[0], t[-1], num_spline_points)

        # Spline degree
        num_points = len(centers)
        if num_points < 4:
            k = num_points - 1  # linear or quadratic
        else:
            k = 3  # cubic

        # Build full spline
        x_spline = make_interp_spline(t, x_points, k=k, bc_type=bc_type)
        y_spline = make_interp_spline(t, y_points, k=k, bc_type=bc_type)

        x_smooth = x_spline(t_new)
        y_smooth = y_spline(t_new)

        # ----------------------------------------------------
        #   Remove points around skip indices
        # ----------------------------------------------------
        skip_idx_set = skip_indices_for_all_paths[path_idx]  # e.g. {3, 5, ...}

        # Convert skip indices -> skip param values in t
        skip_param_values = []
        for i in skip_idx_set:
            if i < len(t):
                skip_t = t[i]
                skip_param_values.append(skip_t)

        # Optionally, define a dynamic gap_radius
        # relative to the total path length:
        total_length = t[-1] if len(t) > 0 else 0
        gap_radius = gap_fraction * total_length

        # Build keep_mask
        keep_mask = np.ones_like(t_new, dtype=bool)
        for skip_t in skip_param_values:
            # We remove all points where |t_new[j] - skip_t| < gap_radius
            keep_mask &= (np.abs(t_new - skip_t) > gap_radius)

        # Final arrays to plot
        t_new_kept = t_new[keep_mask]
        x_kept = x_smooth[keep_mask]
        y_kept = y_smooth[keep_mask]

        # ----------------------------------------------------
        #   Plot the segmented line
        # ----------------------------------------------------
        color = next(color_cycle, 'black')  # fallback if colors exhausted

        if len(t_new_kept) > 1:
            # Identify large jumps > 2 * gap_radius to break lines
            jumps = np.where(np.diff(t_new_kept) > 2 * gap_radius)[0]
            # We'll create segments from [start to jumps[0]],
            # [jumps[0]+1 to jumps[1]], etc.
            start_idx = 0
            for j in jumps:
                seg_slice = slice(start_idx, j+1)
                ax.plot(x_kept[seg_slice], y_kept[seg_slice],
                        color=color, linewidth=2)
                start_idx = j + 1

            # Plot the last segment
            seg_slice = slice(start_idx, len(x_kept))
            ax.plot(x_kept[seg_slice], y_kept[seg_slice],
                    color=color, linewidth=2)

        # ----------------------------------------------------
        #   Plot circles + centers (all original)
        # ----------------------------------------------------
        for (center, radius) in zip(centers, radii):
            c = plt.Circle((center.real, center.imag),
                           radius, fill=False, 
                           edgecolor=color, alpha=0.5)
            ax.add_artist(c)
            # If you want to see center points:
            # ax.plot(center.real, center.imag, 'o', color=color, markersize=5)

    # ----------------------------------------------------
    #   Adjust the axes using a 10% bounding box padding
    # ----------------------------------------------------
    ax.set_aspect('equal', 'box')
    if all_centers:
        all_x = [c.real for c in all_centers]
        all_y = [c.imag for c in all_centers]
        min_x, max_x = np.min(all_x), np.max(all_x)
        min_y, max_y = np.min(all_y), np.max(all_y)
        width = max_x - min_x
        height = max_y - min_y

        # 10% padding in each dimension
        pad_x = padding_fraction * width
        pad_y = padding_fraction * height

        plt.xlim(min_x - pad_x, max_x + pad_x)
        plt.ylim(min_y - pad_y, max_y + pad_y)

    plt.grid(True)
    plt.title(f'Splines With Gaps - {num_spline_points} points')
    # Show or return fig/ax, depending on your needs
    plt.show()

# End Splines Overpass / Underpass _____________________________________________


