"""
Simulation lower energy state, better looking knot
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

import knotpy as kp
from knotpy.drawing.layout import circlepack_layout

from simulation import SimCircle, extract_circle_positions, extract_main_points_and_connections
from simulation import prep_sim, compute_repulsive_forces, limit_displacement, make_step, run_sim, plot_circles

__all__ = ['add_additional_points', 'remove_additional_points', 'run_snap']
__version__ = 'god knows'

# Start of SNAP functions _____________________________________________________

def add_additional_points(ret):
    newret = ret.copy()

    for key, circle in ret.items():
        if isinstance(key, frozenset):
            id_from, id_to = key
            node_from = str(id_from)[0]
            node_to = str(id_to)[0]
            circles_from = ret[key] * ret[node_from]
            circles_to = ret[key] * ret[node_to]
            print(circles_from, circles_to)
            newret[(key, node_from)] = SimCircle(circles_from[0], 0.1)
            newret[(key, node_to)] = SimCircle(circles_to[0], 0.1)

    return newret

def remove_additional_points(ret):
    #newret = ret.copy()
    newret = copy.deepcopy(ret)


    for key, _ in ret.items():
        if isinstance(key, tuple):
            print(key)
            #del newret[key]
            newret.pop(key)

    return newret

def remove_additional_points(ret):
    #newret = ret.copy()
    newret2 = {}


    for key, circle in ret.items():
        if not isinstance(key, tuple):
            print(key)
            #del newret[key]
            #newret.pop(key)
            #newret[key] = circle
            newret2.update({key: circle})


    print("\nnewret2")
    pp.pprint(newret2)
    return newret2

def expand_added_circles(ret):
    print("Expanding circles...")
    #newret = ret.copy()
    newret = copy.deepcopy(ret)

    for key, circle in newret.items():
        if isinstance(key, tuple):
            id_from, id_to = key

            while (circle * ret[id_from]) == []:
                circle.radius += 0.01

            while (circle * ret[id_to]) == []:
                circle.radius += 0.01

    return newret

# Start circle fitting
"""

"""
def circle_from_three_points(p1, p2, p3):
    """
    Compute the circle passing through three non-colinear points.
    Returns the center (h, k) and radius r.
    """
    # Coordinates of the points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculate the determinants
    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2.0
    cd = (temp - x3**2 - y3**2) / 2.0
    det = (x1 - x2)*(y2 - y3) - (x2 - x3)*(y1 - y2)

    if abs(det) < 1e-10:
        raise ValueError("Points are colinear")

    # Center of circle (h, k)
    h = (bc*(y2 - y3) - cd*(y1 - y2)) / det
    k = ((x1 - x2)*cd - (x2 - x3)*bc) / det

    # Radius of circle
    r = np.sqrt((x1 - h)**2 + (y1 - k)**2)

    return h, k, r

def are_points_concyclic(points, tolerance=1e-6):
    """
    Check if all points are concyclic within a specified tolerance.
    Returns True if they are, False otherwise.
    """
    if len(points) <= 3:
        return True  # Three or fewer points are always concyclic

    # Compute circle from first three points
    h, k, r = circle_from_three_points(points[0], points[1], points[2])

    # Check if the rest of the points lie on this circle
    for p in points[3:]:
        x, y = p
        distance = np.sqrt((x - h)**2 + (y - k)**2)
        if abs(distance - r) > tolerance:
            return False
    return True

def circle_through_points(points):
    """
    Compute the circle passing through all given points.
    If the points are concyclic, returns the center and radius.
    Otherwise, raises a ValueError.
    """
    num_points = len(points)
    if num_points < 2:
        raise ValueError("At least two points are required")

    if num_points == 2:
        # Infinite circles pass through two points.
        # Return the circle with center at the midpoint and radius half the distance.
        x1, y1 = points[0]
        x2, y2 = points[1]
        h = (x1 + x2) / 2.0
        k = (y1 + y2) / 2.0
        r = np.sqrt((x1 - h)**2 + (y1 - k)**2)
        return h, k, r

    # For three or more points
    if are_points_concyclic(points):
        # Compute circle from first three points
        h, k, r = circle_from_three_points(points[0], points[1], points[2])
        return h, k, r
    else:
        raise ValueError("Points are not concyclic; no single circle passes through all points")


def fit_circle_least_squares(points):
    """
    Fit a circle to a set of points using least squares minimization.
    Returns the center (h, k) and radius r.
    """
    x = points[:, 0]
    y = points[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f_2, center_estimate)
    xc, yc = center
    Ri       = calc_R(xc, yc)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R


from scipy import optimize

def circle_through_points_or_fit(points):
    """
    Attempts to compute the circle passing through all given points.
    If not possible, fits a circle using least squares.
    Returns the center (h, k) and radius r.
    """
    try:
        # First, try to find an exact circle
        return circle_through_points(points)
    except ValueError:
        # If not possible, fit a circle
        print("Points are not concyclic. Fitting a circle using least squares.")
        h, k, r = fit_circle_least_squares(points)
        return h, k, r
# end circle fitting


def connecting_points(desired_key, ret):
    connection_keys = []
    for key, _ in ret.items():
        if isinstance(key, tuple):
            id_from, id_to = key
            if desired_key in key:
                connection_keys.append(ret[key])
    
    connections = []
    for circle in connection_keys:
        connections.append([circle.center.real, circle.center.imag])
   
    return connections

def run_snap(ret):
    #newret = ret.copy()
    newret = copy.deepcopy(ret)

    for key, _ in newret.items():
        if not isinstance(key, tuple):
            mandatory_points = connecting_points(key, ret)
            h, k, r = circle_through_points_or_fit(mandatory_points)
            #newret[key] = SimCircle(complex(h, k), r)
            newret[key].center = complex(h, k)
            newret[key].radius = r
    
    remove_additional_points(newret)
    return newret

# End of SNAP functions _______________________________________________________

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

    print("\n")
    pp.pprint(ret)
    print("\n")

    filtered_ret = extract_main_points_and_connections(ret)
    pp.pprint(filtered_ret)
    ret1 = add_additional_points(filtered_ret)
    #ret1 = remove_additional_points(ret1)
    #filtered_ret = ret
    #ret1 = run_sim(filtered_ret)
    #ret1 = {key: ret[key].center for key in ret}
    pp.pprint(ret1)
    ret2 = run_sim(ret1)
    pp.pprint(ret2)
    

    plot_circles(filtered_ret, 0)
    plot_circles(ret1, 0)
    plot_circles(ret2, 50)
    #ret3 = expand_added_circles(ret2)
    #plot_circles(ret3, 50)

    #ret4 = run_snap(ret2)
    #pp.pprint(ret4)
    #plot_circles(ret4, 100)

    #ret5 = remove_additional_points(ret4)
    ret5 = remove_additional_points(ret2)
    #ret5 = run_sim(ret5, 3)
    pp.pprint(ret5)
    plot_circles(ret5, 150)
    #draw_from_layout(k2, ret5, with_labels=True, with_title=True)
    plt.plot()