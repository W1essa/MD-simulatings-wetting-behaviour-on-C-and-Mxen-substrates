#!/usr/bin/env python3
import numpy as np
from scipy.spatial import cKDTree
from ase import Atoms       
from ase.build import graphene
import os

# =====================================================================
# --- PARAMETERS ---
# =====================================================================
structure_type = 'flat'  # 'flat' or 'pillars'
target_L = 400.0
pillar_period = 50.0
pillar_radius = 20.0
pillar_height = 20.0
c_c_bond = 1.42
interlayer_dist = 3.35

# =====================================================================
# --- HELPER & GENERATION FUNCTIONS ---
# =====================================================================

def create_base_graphene():
    """Creates a rectangular graphene sheet from a larger rhombic one."""
    a_graphene = c_c_bond * np.sqrt(3)
    nx, ny = int(target_L / a_graphene) + 5, int(target_L / a_graphene) + 5
    base_layer = graphene(size=(nx, ny, 1), vacuum=50.0)
    base_layer.center(axis=(0, 1), vacuum=0.0)
    
    # --- Cut to rectangle ---
    positions = base_layer.get_positions()
    x, y = positions[:, 0], positions[:, 1]
    y_min, y_max = y.min(), y.max()
    tol = 1e-3
    
    mask_ymin = np.abs(y - y_min) < tol
    x_min = x[mask_ymin].min()
    mask_ymax = np.abs(y - y_max) < tol
    x_max = x[mask_ymax].max()

    mask_left = x < x_min - tol
    mask_right = x > x_max + tol
    mask = ~(mask_left | mask_right)
    
    return base_layer[mask]


def create_graphite_stack(base_layer, height, interlayer_dist, c_c_bond):
    """Builds a graphite stack with AB stacking."""
    graphite = base_layer.copy()
    layers_needed = int(np.ceil(height / interlayer_dist))

    for layer_num in range(1, layers_needed):
        new_layer = base_layer.copy()
        if layer_num % 2 == 1: # AB stacking shift
            shift_x = c_c_bond
            shift_y = c_c_bond / np.sqrt(3)
            new_layer.positions[:, 0] += shift_x
            new_layer.positions[:, 1] += shift_y
        new_layer.positions[:, 2] += layer_num * interlayer_dist
        graphite.extend(new_layer)
    return graphite


def generate_pillar_centers(x_range, y_range, radius, period):
    """Generates a grid of pillar centers."""
    def get_centered_coords(min_val, max_val, p, r):
        center = (min_val + max_val) / 2
        span = max_val - min_val - 2 * r
        if span < 0: return np.array([])
        n = int(span / p) + 1
        return center + (np.arange(n) - (n - 1) / 2) * p

    x_centers = get_centered_coords(x_range[0], x_range[1], period, radius)
    y_centers = get_centered_coords(y_range[0], y_range[1], period, radius)
    return [(x, y) for x in x_centers for y in y_centers]


def carve_pillars(graphite_stack, radius, period, base_tol=0.1):
    """Carves pillars from a stack based on generated centers."""
    positions = graphite_stack.get_positions()
    x, y, z = positions.T
    z_min = np.min(z)
    base_mask = np.isclose(z, z_min, atol=base_tol)
    base_positions = positions[base_mask]

    if base_positions.shape[0] == 0: return graphite_stack[base_mask]

    x_range = (np.min(base_positions[:, 0]), np.max(base_positions[:, 0]))
    y_range = (np.min(base_positions[:, 1]), np.max(base_positions[:, 1]))

    pillar_centers = generate_pillar_centers(x_range, y_range, radius, period)
    if not pillar_centers: return graphite_stack[base_mask]
    
    centers_arr = np.array(pillar_centers)
    dist_sq = np.sum((positions[:, np.newaxis, :2] - centers_arr[np.newaxis, :, :])**2, axis=2)
    pillar_mask = np.any(dist_sq <= radius**2, axis=1)
    final_mask = base_mask | pillar_mask
    return graphite_stack[final_mask]

# =====================================================================
# --- VALIDATION & CLEANUP FUNCTION ---
# =====================================================================
# --- constants ---
C_C_MIN = 1.30  # Å
C_C_MAX = 1.60  # Å
C_H     = 1.09  # Å
MIN_SEP = 0.85  # Å safety distance for placed H

# ---------- core utilities ----------
def make_geo(atoms):
    """Build (pos, sym, tree) once to avoid repeated calls."""
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()
    return pos, sym, cKDTree(pos)


def classify_carbon_sites(geo, rmin=C_C_MIN, rmax=C_C_MAX):
    """Classify edge carbons once. Returns:
       edges2: [(i,j,k)] C with 2 C-neighbors in [rmin,rmax]
       edges1: [(i,j)]   C with 1 C-neighbor in [rmin,rmax]"""
    pos, sym, tree = geo
    edges2, edges1 = [], []
    for i, s in enumerate(sym):
        if s != "C":
            continue
        neigh = tree.query_ball_point(pos[i], rmax)
        nb = [j for j in neigh
              if j != i and sym[j] == "C"
              and rmin <= np.linalg.norm(pos[i]-pos[j]) <= rmax]
        if len(nb) == 2:
            edges2.append((i, nb[0], nb[1]))
        elif len(nb) == 1:
            edges1.append((i, nb[0]))
    return edges2, edges1


def passivate_from_classes(atoms, geo, edges2, edges1,
                           ch=C_H, min_sep=MIN_SEP, inplane=True,
                           alpha_deg=90.0):
    """Place H: edges2→1H (outward bisector), edges1→2H (~120° in-plane)."""
    pos, sym, tree = geo
    H_pos = []
    zaxis = np.array([0.0, 0.0, 1.0])

    def unit(v):
        if inplane:
            v = v.copy(); v[2] = 0.0
        n = np.linalg.norm(v)
        return (v / n) if n > 1e-8 else None

    def place_from(pC, d):
        pH = pC + ch * d
        dmin, _ = tree.query(pH, k=1)
        if (np.isscalar(dmin) and dmin < min_sep) or (not np.isscalar(dmin) and min(dmin) < min_sep):
            pH = pC + (ch + 0.2) * d
        return pH

    # 2-neighbor sites: 1 H along outward bisector
    for i, j, k in edges2:
        pC = pos[i]; pN = 0.5 * (pos[j] + pos[k])
        d = unit(pC - pN)
        if d is not None:
            H_pos.append(place_from(pC, d))

        # 1-neighbor sites: 2 H at a tunable angle alpha from the outward normal
    alpha = np.deg2rad(alpha_deg)
    ca, sa = np.cos(alpha), np.sin(alpha)

    for i, j in edges1:
        pC = pos[i]
        v1 = unit(pC - pos[j])           # outward normal (in-plane if inplane=True)
        if v1 is None:
            continue

        # in-plane perpendicular to v1
        u = np.cross(zaxis, v1)
        if np.linalg.norm(u) < 1e-6:
            u = np.cross([1.0, 0.0, 0.0], v1)
        u /= np.linalg.norm(u)

        # two directions symmetric around v1 by ±alpha
        for d in (ca * v1 + sa * u, ca * v1 - sa * u):
            d = unit(d)
            if d is not None:
                H_pos.append(place_from(pC, d))


    if not H_pos:
        return atoms
    Hs = Atoms('H' * len(H_pos), positions=np.array(H_pos))
    out = atoms.copy()
    out.extend(Hs)
    return out


def passivate_edges(atoms, ch=C_H, min_sep=MIN_SEP, inplane=True, alpha_deg=60.0):
    geo = make_geo(atoms)
    edges2, edges1 = classify_carbon_sites(geo)
    return passivate_from_classes(atoms, geo, edges2, edges1,
                                  ch=ch, min_sep=min_sep, inplane=inplane,
                                  alpha_deg=alpha_deg)



def overlaps(geo, cutoff=0.9):
    """Unique close-contact pairs (i<j) with distance < cutoff Å."""
    pos, _sym, tree = geo  # _sym is unused; kept for a uniform (pos, sym, tree) tuple
    pairs = tree.query_pairs(cutoff)  # set of (i, j) with i<j
    if not pairs:
        return []
    return [(i, j, float(np.linalg.norm(pos[i] - pos[j]))) for (i, j) in sorted(pairs)]


# =====================================================================
# --- FILE WRITING FUNCTION ---
# =====================================================================
def write_lammps_full(structure: Atoms, filename: str, padding: float = 20.0, reserve_water_types: bool = False):
    """
    Write LAMMPS data in 'full' style from an ASE Atoms object.

    Atom Types:
    - 1: C (Carbon)
    - 2: H (Hydrogen)
    - 3: O (Oxygen, reserved for water)
    - 4: H (Hydrogen, reserved for water)

    Args:
        structure (ase.atoms.Atoms): The input ASE structure.
        filename (str): The output .data file path.
        padding (float): Extra space (Angstroms) to add to the box boundaries.
        reserve_water_types (bool): If True, reserves types 3 (O) and 4 (H)
                                    and sets the header to 4 atom types.
                                    
                                    !!! FIX: Set to False by default.
                                    We only want C and H (types 1, 2)
                                    for the initial graphene file.
    """
    pos = structure.get_positions()
    sym = structure.get_chemical_symbols()
    
    # --- Graphene Atom Type Mapping ---
    tmap = {"C": 1, "H": 2}
    types = [tmap.get(s, 1) for s in sym]

    # --- Box Dimensions ---
    xlo, ylo, zlo = pos.min(axis=0) - padding
    xhi, yhi, zhi = pos.max(axis=0) + padding

    # --- Atom Type Count (CRITICAL FIX) ---
    max_graphene_type = 1
    if 2 in types:
        max_graphene_type = 2

    # Set the total number of types for the header.
    if reserve_water_types:
        # (e.g., if merging manually)
        ntypes = 4
    else:
        # (Correct for graphene-only file)
        ntypes = max_graphene_type

    # --- File Writing ---
    with open(filename, "w") as f:
        # Header
        f.write("LAMMPS data: graphene (full)\n\n")
        f.write(f"{len(pos)} atoms\n\n")
        f.write(f"{ntypes} atom types\n\n") # This will now correctly write 2
        
        # Box
        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")

        # Masses
        f.write("Masses\n\n")
        f.write("1 12.011\n") # C
        if max_graphene_type == 2:
            f.write("2 1.008\n")  # H (graphene)
        
        if reserve_water_types:
            f.write("3 15.9994\n") # O (water)
            f.write("4 1.008\n")    # H (water)
        f.write("\n")

        # Atoms section
        # Format: id mol-id type charge x y z
        f.write("Atoms # full\n\n")
        for i, (t, (x, y, z)) in enumerate(zip(types, pos), start=1):
            # Set mol-id to 1 (needed for atom_style full)
            f.write(f"{i} 1 {t} 0.0 {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"✅ LAMMPS data (full, graphene-only) written to {filename}")



# =====================================================================
# --- MAIN EXECUTION BLOCK ---
# =====================================================================
if __name__ == "__main__":
    # --- I/O setup ---
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()
    output_dir = os.path.join(os.path.dirname(BASE_DIR), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_data_file = os.path.join(output_dir, f"graphene_{structure_type}_initial.data")

    # --- Build base sheet ---
    base = create_base_graphene()

    # --- Carbon scaffold (flat or pillars) ---
    if structure_type == "flat":
        carbon = base
    elif structure_type == "pillars":
        stack  = create_graphite_stack(base, pillar_height, interlayer_dist, c_c_bond)
        carbon = carve_pillars(stack, pillar_radius, pillar_period)
    else:
        raise ValueError("structure_type must be 'flat' or 'pillars'")

    # --- Edge passivation (C → CH) ---
    gH = passivate_edges(carbon, ch=C_H, min_sep=MIN_SEP, inplane=True)

    # --- Quick sanity check (overlaps) ---
    pos, sym, tree = make_geo(gH)
    if overlaps((pos, sym, tree), cutoff=0.90):
        print("[WARN] close contacts detected; tune MIN_SEP or geometry.")

    # --- Write LAMMPS data (full; 1=C, 2=H) ---
    write_lammps_full(gH, output_data_file)

    # --- Report geometry (no padding) ---
    positions = gH.get_positions()
    mins = np.min(positions, axis=0)
    maxs = np.max(positions, axis=0)
    dims = maxs - mins
    print("\n--- Graphene Structure Dimensions ---")
    print(f"X-range: {mins[0]:.4f} to {maxs[0]:.4f} Å")
    print(f"Y-range: {mins[1]:.4f} to {maxs[1]:.4f} Å")
    print(f"Z-range: {mins[2]:.4f} to {maxs[2]:.4f} Å")
    print(f"Dimensions: {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f} Å")
    print("-------------------------------------")