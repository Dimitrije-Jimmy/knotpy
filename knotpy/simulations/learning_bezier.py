import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly

# Dictionary of points with keys as 'a', 'b', etc., and values as complex numbers representing x and y
points_dict = {
    'a': complex(0, 0),
    'b': complex(1, 2),
    'c': complex(3, 3),
    'd': complex(4, 0),
    'e': complex(2, -2)
}

# Imports and funcs for PD ________________________________________________________

from knotpy.drawing.circlepack import CirclePack
from knotpy.drawing.draw_matplotlib import draw_from_layout
from knotpy.notation.native import from_knotpy_notation
from knotpy.notation.pd import from_pd_notation
from knotpy.drawing.layout import circlepack_layout

def extract_circle_positions(circle_dict):
    """Extract circle centers from the dictionary of circles."""
    positions = {}
    for key, circle in circle_dict.items():
        positions[key] = circle.center  # Each circle has a .center attribute
    return positions

def extract_main_points_and_connections(ret):
    return {k: v for k, v in ret.items() if isinstance(k, (str, frozenset))}

# End of imports ________________________________________________________________

def extract_points(points_dict, keys):
    """Extracts x and y coordinates from the points dictionary based on the provided keys."""
    points = [points_dict[key] for key in keys]
    x = [p.real for p in points]
    y = [p.imag for p in points]
    return x, y

def bernstein_poly(n, k, t):
    """Calculates the Bernstein polynomial of n, k as a function of t."""
    from scipy.special import comb
    return comb(n, k) * (t ** k) * ((1 - t) ** (n - k))

def bezier_curve_custom(points_dict, keys, num=200):
    """Generates a Bézier curve using custom implementation."""
    x_points, y_points = extract_points(points_dict, keys)
    n = len(x_points) - 1
    t = np.linspace(0, 1, num)
    curve_x = np.zeros_like(t)
    curve_y = np.zeros_like(t)
    for k in range(n + 1):
        bern_poly = bernstein_poly(n, k, t)
        curve_x += x_points[k] * bern_poly
        curve_y += y_points[k] * bern_poly
    return curve_x, curve_y

def bezier_curve_scipy(points_dict, keys, num=200):
    """Generates a Bézier curve using scipy's BPoly."""
    x_points, y_points = extract_points(points_dict, keys)
    t = np.linspace(0, 1, num)
    # Coefficients need to be of shape (number of intervals, degree + 1)
    # Since we have one interval from 0 to 1, the number of intervals k = 1
    # So coefficients should be of shape (1, degree + 1)
    c_x = np.array([x_points])  # Shape (1, n+1)
    c_y = np.array([y_points])  # Shape (1, n+1)
    x = np.array([0, 1])  # Breakpoints
    # Create BPoly
    bezier_x = BPoly(c_x, x)(t)
    bezier_y = BPoly(c_y, x)(t)
    return bezier_x, bezier_y

def plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Bézier Curve", loop=False):
    """Plots the Bézier curve along with control points."""
    plt.figure(figsize=(8, 6))
    # Plot control points
    plt.plot(x_points, y_points, 'ro--', label='Control Points')
    # Plot Bézier curve
    plt.plot(curve_x, curve_y, 'b-', label='Bézier Curve', linewidth=2)
    # If loop, handle overlapping lines
    if loop:
        # Split the curve into segments where the top line can pass over the bottom
        mid_index = len(curve_x) // 2
        plt.plot(curve_x[:mid_index], curve_y[:mid_index], 'b-', linewidth=2)
        plt.plot(curve_x[mid_index:], curve_y[mid_index:], 'b-', linewidth=2)
        plt.plot(curve_x[mid_index - 10:mid_index + 10], curve_y[mid_index - 10:mid_index + 10], 'w', linewidth=4)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Example usage:

# Linear Bézier Curve (2 points)
keys_linear = ['a', 'b']
x_points, y_points = extract_points(points_dict, keys_linear)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_linear)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Linear Bézier Curve")

# Quadratic Bézier Curve (3 points)
keys_quadratic = ['a', 'b', 'c']
x_points, y_points = extract_points(points_dict, keys_quadratic)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_quadratic)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Quadratic Bézier Curve")

# Cubic Bézier Curve (4 points)
keys_cubic = ['a', 'b', 'c', 'd']
x_points, y_points = extract_points(points_dict, keys_cubic)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_cubic)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Cubic Bézier Curve")

# Quartic Bézier Curve (5 points)
keys_quartic = ['a', 'b', 'c', 'd', 'e']
x_points, y_points = extract_points(points_dict, keys_quartic)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_quartic)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Quartic Bézier Curve")

# Bézier Curve with Loop
keys_loop = ['a', 'b', 'e', 'd', 'c']
x_points, y_points = extract_points(points_dict, keys_loop)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_loop)
#plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Bézier Curve with Loop", loop=True)

# Using scipy's BPoly
x_points, y_points = extract_points(points_dict, keys_cubic)
#curve_x_scipy, curve_y_scipy = bezier_curve_scipy(points_dict, keys_cubic)
#plot_bezier_curve(curve_x_scipy, curve_y_scipy, x_points, y_points, title="Cubic Bézier Curve (scipy)")

# Quintic Bézier Curve (6 points)
points_dict['f'] = complex(5, 2)
keys_quintic = ['a', 'b', 'c', 'd', 'e', 'f']
x_points, y_points = extract_points(points_dict, keys_quintic)
curve_x, curve_y = bezier_curve_custom(points_dict, keys_quintic)
plot_bezier_curve(curve_x, curve_y, x_points, y_points, title="Quintic Bézier Curve")

plt.show()

# Using matplotlib.path and matplotlib.patches ______________________________________________

import matplotlib.patches as patches
from matplotlib.path import Path

def bezier_curve_matplotlib(points_dict, keys):
    """Generates a Bézier curve using matplotlib's Path."""
    points = [points_dict[key] for key in keys]
    verts = [(p.real, p.imag) for p in points]
    if len(verts) == 4:
        codes = [Path.MOVETO,
                 Path.CURVE4,
                 Path.CURVE4,
                 Path.CURVE4]
    elif len(verts) == 3:
        codes = [Path.MOVETO,
                 Path.CURVE3,
                 Path.CURVE3]
    else:
        raise ValueError("Matplotlib's Path supports quadratic (3 points) and cubic (4 points) Bézier curves.")
    path = Path(verts, codes)
    return path

def plot_bezier_curve_matplotlib(path, x_points, y_points, title="Bézier Curve"):
    """Plots the Bézier curve using matplotlib's PathPatch."""
    fig, ax = plt.subplots()
    patch = patches.PathPatch(path, facecolor='none', lw=2, edgecolor='blue')
    ax.add_patch(patch)
    ax.plot(x_points, y_points, 'ro--', label='Control Points')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.show()

# Example usage:
keys_cubic = ['a', 'b',  'd']
x_points, y_points = extract_points(points_dict, keys_cubic)
path = bezier_curve_matplotlib(points_dict, keys_cubic)
plot_bezier_curve_matplotlib(path, x_points, y_points, title="Cubic Bézier Curve (matplotlib)")


