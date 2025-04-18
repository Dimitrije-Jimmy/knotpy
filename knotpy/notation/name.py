"""Conway names (3_1, 4_1, ...).

For knots with 10 or fewer crossings we use the classical names (polished by Perko). For knots with 11 crossings, the
naming convention is that of Dowker-Thistlethwaite.
"""

from pathlib import Path
import csv

from knotpy.notation.pd import from_pd_notation
from knotpy.classes.planardiagram import PlanarDiagram
from knotpy.reidemeister.simplification import simplify_diagram_crossing_reducing
from knotpy.algorithms.structure import is_knot

also_known_as = {
    "unknot": "0_1", "trefoil": "3_1", "figure eight": "4_1", "figure 8": "4_1", "cinquefoil": "5_1",
    "pentafoil": "5_1", "3-twist": "5_2", "stevedore": "6_1", "miller institute": "6_2", "septafoil": "7_1",
    "nonafoil": "9_1"
}

database_root_folder = Path("./database")
path_PD_up_to_10_crossings = database_root_folder / "PD_knots_up_to_10_crossings.csv"

_knots_up_to_10_crossings = None  # contains dictionaty, e.g. {"3_1":  {'PD': '[[1;5;2;4];[3;1;4;6];[5;3;6;2]]', 'symmetry': 'reversible'}, ...}

def _name_unoriented_knot(k: PlanarDiagram):
    global _knots_up_to_10_crossings
    global path_PD_up_to_10_crossings

    if _knots_up_to_10_crossings is None:
        pass

    k = simplify_diagram_crossing_reducing(k)
    print(k)

def name(k: PlanarDiagram):
    return None

def from_name(name):
    global _knots_up_to_10_crossings
    global path_PD_up_to_10_crossings

    name = name.strip().lower()

    # common names
    if name in also_known_as:
        name = also_known_as[name]

    if "a" in name or "n" in name:
        raise NotImplementedError("loading knots with more than 10 crossings not yet implemented")

    if name.find("_") == -1:
        raise ValueError("Classical (Conway) knot name not in correct format, should be e.g. \"3_1\".")

    try:
        n = int(name[:name.find("_")])  # number of crossings
    except ValueError:
        raise ValueError(f"cannot find a knot with name {name} (name should start with an integer, e.g. 3_1)")

    if n > 10:
        raise ValueError("the KnotPy database only contains knots up to 10 crossings")

    if _knots_up_to_10_crossings is None:
        # preload knot names
        _knots_up_to_10_crossings = dict()
        with open(path_PD_up_to_10_crossings, 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                _knots_up_to_10_crossings[line["name"]] = {"PD": line["PD"], "symmetry": line["symmetry"]}

    if name not in _knots_up_to_10_crossings:
        raise ValueError(f"cannot find a knot with name {name} in the KnotPy database")

    pd_code = _knots_up_to_10_crossings[name]["PD"]
    symmetry = _knots_up_to_10_crossings[name]["symmetry"]
    k = from_pd_notation(text=pd_code, create_using=Knot, name=name, symmetry=symmetry)
    return k


if __name__ == "__main__":
    print(from_name("7_1"))
    print(from_name("trefoil"))
    print(from_name("figure eight"))

    """
    predicted file sizes:
    
    16 knots: n = 1,701,934
    
    PD: 
    [(10,0,11,31),(0,14,1,13),(14,2,15,1),(23,3,24,2), ....
    20 bitov * 16 = 320 bytes per knot = 68 MB
    
    DT:
    [4, 8, 10, 14, 2, 16, 20, 6, 22, 12, 24, 18] -> [2,4,5,7,1,8,10,3,11,6,12,9] -> 0,...
    DT: 12 * 5 = 60 bitov 0 =  102 MB
    
    
    """