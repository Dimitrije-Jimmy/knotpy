import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import LineString
import math
import math

# Helper function to convert polar coordinates to Cartesian
def polar_to_cartesian(angle_deg, radius):
    angle_rad = np.deg2rad(angle_deg)
    x = radius * np.cos(angle_rad)
    y = radius * np.sin(angle_rad)
    return x, y

# Main points
P0 = (0, 2)
P1 = polar_to_cartesian(210, 2)  # Angle 210 degrees, radius 2
P2 = polar_to_cartesian(-30, 2)  # Angle -30 degrees, radius 2

# Control points
# First Bezier curve from P0 to P1
# Control points C0 and C1
C0 = (P0[0] + 2.2, P0[1] + 0)  # +(2.2, 0)
delta_x1, delta_y1 = polar_to_cartesian(120, -2.2)
C1 = (P1[0] + delta_x1, P1[1] + delta_y1)

# Second Bezier curve from P1 to P2
delta_x2, delta_y2 = polar_to_cartesian(120, 2.2)
C2 = (P1[0] + delta_x2, P1[1] + delta_y2)
delta_x3, delta_y3 = polar_to_cartesian(60, 2.2)
C3 = (P2[0] + delta_x3, P2[1] + delta_y3)

# Third Bezier curve from P2 back to P0
delta_x4, delta_y4 = polar_to_cartesian(60, -2.2)
C4 = (P2[0] + delta_x4, P2[1] + delta_y4)
C5 = (P0[0] - 2.2, P0[1] + 0)  # +(-2.2, 0)

from matplotlib.path import Path

# Vertices
vertices = [
    P0,     # MOVETO
    C0, C1, P1,  # First CURVE4
    C2, C3, P2,  # Second CURVE4
    C4, C5, P0,  # Third CURVE4
    (0, 0)  # Placeholder for CLOSEPOLY
]

# Path codes
codes = [
    Path.MOVETO,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CLOSEPOLY
]

import matplotlib.pyplot as plt

# Create the Path
path = Path(vertices, codes)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

# Add the Path as a Patch
patch = patches.PathPatch(path, facecolor='none', lw=2)
ax.add_patch(patch)

# Set plot limits
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_title("Trefoil Knot without Over/Under Crossings")
ax.grid(True)

plt.show()


####
# Helper functions
def polar_to_cartesian(angle_deg, radius):
    angle_rad = np.deg2rad(angle_deg)
    x = radius * np.cos(angle_rad)
    y = radius * np.sin(angle_rad)
    return x, y

def bezier_curve(p0, c0, c1, p1, t):
    return ((1 - t) ** 3) * np.array(p0) + \
           3 * ((1 - t) ** 2) * t * np.array(c0) + \
           3 * (1 - t) * t ** 2 * np.array(c1) + \
           t ** 3 * np.array(p1)

def find_self_intersections(line):
    coords = np.array(line.coords)
    intersections = []
    for i in range(len(coords) - 1):
        seg1 = LineString([coords[i], coords[i+1]])
        for j in range(i+2, len(coords) - 1):
            seg2 = LineString([coords[j], coords[j+1]])
            if seg1.crosses(seg2):
                intersection_point = seg1.intersection(seg2)
                intersections.append((i, j, intersection_point))
    return intersections

# Define the points and control points
P0 = (0, 2)
P1 = polar_to_cartesian(210, 2)
P2 = polar_to_cartesian(-30, 2)

# Control points
C0 = (P0[0] + 2.2, P0[1] + 0)
delta_x1, delta_y1 = polar_to_cartesian(120, -2.2)
C1 = (P1[0] + delta_x1, P1[1] + delta_y1)

delta_x2, delta_y2 = polar_to_cartesian(120, 2.2)
C2 = (P1[0] + delta_x2, P1[1] + delta_y2)
delta_x3, delta_y3 = polar_to_cartesian(60, 2.2)
C3 = (P2[0] + delta_x3, P2[1] + delta_y3)

delta_x4, delta_y4 = polar_to_cartesian(60, -2.2)
C4 = (P2[0] + delta_x4, P2[1] + delta_y4)
C5 = (P0[0] - 2.2, P0[1] + 0)

# Evaluate the Bezier curves
num_points = 1000
t_values = np.linspace(0, 1, num_points)
curve1 = bezier_curve(P0, C0, C1, P1, t_values)
curve2 = bezier_curve(P1, C2, C3, P2, t_values)
curve3 = bezier_curve(P2, C4, C5, P0, t_values)
knot_curve = np.vstack((curve1, curve2, curve3))

# Find self-intersections
knot_line = LineString(knot_curve)
intersections = find_self_intersections(knot_line)

# Get the indices of the intersection points
split_indices = []
for i, j, point in intersections:
    split_indices.extend([i, j])

split_indices = sorted(set(split_indices))

# Split the curve at the intersection points
segments = []
prev_index = 0
for index in split_indices:
    segments.append(knot_curve[prev_index:index+1])
    prev_index = index+1
segments.append(knot_curve[prev_index:])

# Decide which segments to have gaps (indices start from 0)
gap_segments = [1, 3, 5]  # Adjust based on the intersections

# Plot the trefoil knot with over/under crossings
fig, ax = plt.subplots(figsize=(8, 8))

for i, segment in enumerate(segments):
    if i in gap_segments:
        # Skip plotting this segment to create a gap
        continue
    ax.plot(segment[:, 0], segment[:, 1], color='blue', lw=2)

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_title("Trefoil Knot with Over/Under Crossings")
ax.grid(True)

plt.show()
