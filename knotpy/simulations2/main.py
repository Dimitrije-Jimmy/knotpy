"""
Main Dev script, utilising subsidiaries
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

from spline import plot_spline, build_circles_data, extract_points
from spline import plot_bezier_curve_matplotlib,bezier_curve_matplotlib, bezier_curve_matplotlib_multiple, get_positions

from circlesnap import add_additional_points, remove_additional_points, run_snap

from sequence import process_paths, process_path, plot_splines, build_all_circles_data

from circlesnap import add_additional_points, remove_additional_points
from complex_sim import run_complex_sim
from networkX import plot_networkx_layout, plot_networkx_layout2

#__all__ = ['process_paths', 'process_path', 'plot_splines', 'build_all_circles_data']
__version__ = 'god knows'
__author__ = 'Dimitrije Pešić'



# Let's play __________________________________________________________________

# Trefoil:
s = "X[4,2,5,1],X[2,6,3,5],X[6,4,1,3]"
# Large:
#s = "X[1,9,2,8],X[3,10,4,11],X[5,3,6,2],X[7,1,8,12],X[9,4,10,5],X[11,7,12,6]"
#s = "V[0,1,2], V[0,2,1]"
#s = "V[1,2,0], V[1,3,4], X[5,6,3,0], X[7,5,2,8], X[6,7,8,4]"
# Problematic:
#s = "X[0,1,2,3],X[4,5,6,7],X[1,8,9,10],X[11,12,13,9],X[14,15,7,16],X[17,18,19,13],X[10,20,17,12],X[20,19,15,14],V[3,21,22],V[5,23,24],V[6,24,16],V[4,18,23],V[0,22,8],V[2,11,21]"
k2 = from_pd_notation(s)
print(k2)
ret = circlepack_layout(k2)
ret = extract_main_points_and_connections(ret)
pp.pprint(ret)

ret1 = run_sim(ret, 100)    # just simulation
ret2 = add_additional_points(ret)
ret3 = run_sim(ret2, 100)
ret4 = remove_additional_points(ret3)

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
all_circles_data3 = build_all_circles_data(processed_paths, ret4)
pp.pprint(all_circles_data2)


# Plot all splines on the same figure
plot_splines(all_circles_data, 300)
plot_splines(all_circles_data, 100)
plot_splines(all_circles_data, 1000)
plot_splines(all_circles_data2, 1000)
plot_splines(all_circles_data3, 1000)
plt.show()