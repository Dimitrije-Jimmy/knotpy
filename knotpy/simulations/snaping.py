import numpy as np

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


# Example usage:
if __name__ == "__main__":
    # Define your mandatory points
    mandatory_points = np.array([
        [1, 2],
        [4, 6],
        [5, 2],
        #[3, 5]
    ])

    # Try to find an exact circle
    try:
        h, k, r = circle_through_points(mandatory_points)
        print(f"Exact circle found:")
        print(f"Center: ({h}, {k})")
        print(f"Radius: {r}")
    except ValueError:
        # If no exact circle exists, fit a circle
        h, k, r = fit_circle_least_squares(mandatory_points)
        print(f"No exact circle found. Circle fitted using least squares:")
        print(f"Center: ({h}, {k})")
        print(f"Radius: {r}")


exit()
