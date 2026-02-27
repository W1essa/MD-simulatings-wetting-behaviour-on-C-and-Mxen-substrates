#!/usr/bin/env python3
import os
import numpy as np
from scipy.spatial import cKDTree
from ase import Atoms
from ase.build import graphene

# =====================================================================
# --- CONFIGURATION PARAMETERS ---
# =====================================================================
STRUCTURE_TYPE = 'pillars'  # Options: 'flat' or 'pillars'
TARGET_L = 400.0            # Sheet size in Angstroms
PILLAR_PERIOD = 50.0        # Distance between pillars
PILLAR_RADIUS = 20.0        # Radius of the carved pillars
PILLAR_HEIGHT = 20.0        # Height of the stack
C_C_BOND = 1.42             # Carbon-Carbon bond length
INTERLAYER_DIST = 3.35      # Graphite interlayer distance

# Passivation constants
C_C_MIN = 1.30              # Minimum valid C-C bond length
C_C_MAX = 1.60              # Maximum valid C-C bond length
C_H = 1.09                  # Standard C-H bond length
MIN_SEP = 0.85              # Minimum steric distance for H placement

# =====================================================================
# --- GENERATION FUNCTIONS ---
# =====================================================================
def create_base_graphene():
    """Create a rectangular monolayer graphene sheet."""
    a_graphene = C_C_BOND * np.sqrt(3)
    nx = int(TARGET_L / a_graphene) + 5
    ny = int(TARGET_L / a_graphene) + 5
    
    base_layer = graphene(size=(nx, ny, 1), vacuum=50.0)
    base_layer.center(axis=(0, 1), vacuum=0.0)
    
    # Cut to a precise rectangle
    positions = base_layer.get_positions()
    x, y, z = positions.T
    y_min, y_max = y.min(), y.max()
    tol = 1e-3
    
    mask_ymin = np.abs(y - y_min) < tol
    x_min = x[mask_ymin].min()
    
    mask_ymax = np.abs(y - y_max) < tol
    x_max = x[mask_ymax].max()

    mask_left = x < x_min - tol
    mask_right = x > x_max + tol
    valid_mask = ~(mask_left | mask_right)
    
    return base_layer[valid_mask]

def create_graphite_stack(base_layer, height, interlayer_dist, bond_length):
    """Build an AB-stacked graphite structure from a base layer."""
    graphite = base_layer.copy()
    layers_needed = int(np.ceil(height / interlayer_dist))

    for layer_num in range(1, layers_needed):
        new_layer = base_layer.copy()
        # Apply AB stacking shift for odd layers
        if layer_num % 2 == 1: 
            shift_x = bond_length
            shift_y = bond_length / np.sqrt(3)
            new_layer.positions[:, 0] += shift_x
            new_layer.positions[:, 1] += shift_y
            
        new_layer.positions[:, 2] += layer_num * interlayer_dist
        graphite.extend(new_layer)
        
    return graphite

def generate_pillar_centers(x_range, y_range, radius, period):
    """Calculate grid center coordinates for pillar carving."""
    def get_centered_coords(min_val, max_val, p, r):
        center = (min_val + max_val) / 2.0
        span = max_val - min_val - 2.0 * r
        if span < 0:
            return np.zeros(0)  # Safe return to avoid empty list errors
        n = int(span / p) + 1
        return center + (np.arange(n) - (n - 1) / 2.0) * p

    x0, x1 = x_range
    y0, y1 = y_range
    x_centers = get_centered_coords(x0, x1, period, radius)
    y_centers = get_centered_coords(y0, y1, period, radius)
    
    return [(x, y) for x in x_centers for y in y_centers]

def carve_pillars(graphite_stack, radius, period, base_tol=0.1):
    """Carve cylindrical pillars out of the graphite stack."""
    positions = graphite_stack.get_positions()
    x, y, z = positions.T
    z_min = np.min(z)
    
    # Keep the entire bottom layer intact
    base_mask = np.isclose(z, z_min, atol=base_tol)
    base_positions = positions[base_mask]

    if base_positions.shape == 0:
        return graphite_stack[base_mask]

    x_range = (np.min(base_positions[:, 0]), np.max(base_positions[:, 0]))
    y_range = (np.min(base_positions[:, 1]), np.max(base_positions[:, 1]))

    pillar_centers = generate_pillar_centers(x_range, y_range, radius, period)
    if not pillar_centers:
        return graphite_stack[base_mask]
    
    centers_arr = np.array(pillar_centers)
    
    # Calculate squared distances in the XY plane
    xy_positions = positions[:, np.newaxis, :2]
    dist_sq = np.sum((xy_positions - centers_arr[np.newaxis, :, :])**2, axis=2)
    
    # Keep atoms within the radius of any pillar center
    pillar_mask = np.any(dist_sq <= radius**2, axis=1)
    final_mask = base_mask | pillar_mask
    
    return graphite_stack[final_mask]

# =====================================================================
# --- PASSIVATION & CLEANUP FUNCTIONS ---
# =====================================================================
def clean_dangling_carbons(atoms, rmin=C_C_MIN, rmax=C_C_MAX):
    """Iteratively remove unstable 0- and 1-coordinated Carbon atoms."""
    clean_atoms = atoms.copy()
    while True:
        pos = clean_atoms.get_positions()
        tree = cKDTree(pos)
        to_keep = list()
        
        for i, pC in enumerate(pos):
            idx = tree.query_ball_point(pC, rmax)
            neighbors = [
                j for j in idx 
                if j!= i and rmin <= np.linalg.norm(pC - pos[j]) <= rmax
            ]
            # Keep atoms with at least 2 stable bonds
            if len(neighbors) >= 2:
                to_keep.append(i)
        
        # Stop if the structure has stabilized
        if len(to_keep) == len(pos):
            break
        clean_atoms = clean_atoms[to_keep]
        
    return clean_atoms

def make_geo(atoms):
    """Generate geometry data (positions, symbols, KDTree)."""
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()
    return pos, sym, cKDTree(pos)

def classify_carbon_sites(geo, rmin=C_C_MIN, rmax=C_C_MAX):
    """Classify edge carbons by counting valid neighbors."""
    pos, sym, tree = geo
    edges2 = list()  # Stores tuples: (atom_index, neighbor_0_index, neighbor_1_index)
    edges1 = list()  # Stores tuples: (atom_index, neighbor_0_index)
    
    for i, symbol in enumerate(sym):
        if symbol != "C":
            continue
            
        neighbors_idx = tree.query_ball_point(pos[i], rmax)
        valid_bonds = [
            j for j in neighbors_idx 
            if j != i and sym[j] == "C" and rmin <= np.linalg.norm(pos[i] - pos[j]) <= rmax
        ]
        
        # Extract exact indices to prevent IndexError and enforce tuple structure
        if len(valid_bonds) == 2:
            edges2.append((i, valid_bonds[0], valid_bonds[1]))
        elif len(valid_bonds) == 1:
            edges1.append((i, valid_bonds[0]))
            
    return edges2, edges1

def passivate_from_classes(atoms, geo, edges2, edges1, ch=C_H, min_sep=MIN_SEP, inplane=True, alpha_deg=90.0):
    """Passivate identified edge sites with Hydrogen atoms."""
    pos, sym, tree = geo
    h_positions = list()
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Track all positions dynamically to prevent atomic overlaps
    dynamic_pos = list(pos)

    def unit_vector(v):
        """Normalize a vector, optionally restricting to the XY plane."""
        if inplane:
            v = np.copy(v)
            v[2] = 0.0
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 1e-8 else None

    def place_hydrogen(pC, direction):
        """Calculate optimal Hydrogen placement considering steric hindrance."""
        pH = pC + ch * direction
        temp_tree = cKDTree(dynamic_pos)
        dmin, _ = temp_tree.query(pH, k=1)
        
        min_dist = dmin if np.isscalar(dmin) else np.min(dmin)
            
        # Push outward if the standard distance causes a clash
        if min_dist < min_sep:
            pH = pC + (ch + 0.3) * direction  
        
        dynamic_pos.append(pH)
        return pH

    # Passivate sites with 2 neighbors (1 Hydrogen along the outward bisector)
    for i, j, k in edges2:
        pC = pos[i]
        pN = 0.5 * (pos[j] + pos[k])
        direction = unit_vector(pC - pN)
        if direction is not None:
            h_positions.append(place_hydrogen(pC, direction))

    # Passivate sites with 1 neighbor (2 Hydrogens at specific angles)
    alpha = np.deg2rad(alpha_deg)
    ca, sa = np.cos(alpha), np.sin(alpha)

    for i, j in edges1:
        pC = pos[i]
        v1 = unit_vector(pC - pos[j])
        if v1 is None:
            continue

        # Calculate perpendicular vector in the plane
        u = np.cross(z_axis, v1)
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(np.array([1.0, 0.0, 0.0]), v1)
        u /= np.linalg.norm(u)

        # Place two hydrogens symmetrically
        for d in (ca * v1 + sa * u, ca * v1 - sa * u):
            direction = unit_vector(d)
            if direction is not None:
                h_positions.append(place_hydrogen(pC, direction))

    if not h_positions:
        return atoms
        
    hydrogen_atoms = Atoms('H' * len(h_positions), positions=np.array(h_positions))
    final_structure = atoms.copy()
    final_structure.extend(hydrogen_atoms)
    return final_structure

def passivate_edges(atoms, ch=C_H, min_sep=MIN_SEP, inplane=True, alpha_deg=60.0):
    """Master function to clean and passivate graphene edges."""
    cleaned_atoms = clean_dangling_carbons(atoms)
    geo = make_geo(cleaned_atoms)
    edges2, edges1 = classify_carbon_sites(geo)
    return passivate_from_classes(
        cleaned_atoms, geo, edges2, edges1,
        ch=ch, min_sep=min_sep, inplane=inplane, alpha_deg=alpha_deg
    )

def check_overlaps(geo, cutoff=0.9):
    """Detect unphysical atomic overlaps."""
    pos, _sym, tree = geo
    pairs = tree.query_pairs(cutoff)
    if not pairs:
        return list()
    return [(i, j, float(np.linalg.norm(pos[i] - pos[j]))) for (i, j) in sorted(pairs)]

# =====================================================================
# --- FILE WRITING FUNCTION ---
# =====================================================================
def write_lammps_full(structure: Atoms, filename: str, padding: float = 20.0):
    """Export the structure to LAMMPS 'full' data format."""
    pos = structure.get_positions()
    sym = structure.get_chemical_symbols()
    
    type_map = {"C": 1, "H": 2}
    atom_types = [type_map.get(s, 1) for s in sym]
    num_types = 2 if 2 in atom_types else 1

    mins = pos.min(axis=0) - padding
    maxs = pos.max(axis=0) + padding

    with open(filename, "w") as f:
        f.write("LAMMPS data: graphene (full)\n\n")
        f.write(f"{len(pos)} atoms\n\n")
        f.write(f"{num_types} atom types\n\n")
        
        f.write(f"{mins[0]:.6f} {maxs[0]:.6f} xlo xhi\n")
        f.write(f"{mins[1]:.6f} {maxs[1]:.6f} ylo yhi\n")
        f.write(f"{mins[2]:.6f} {maxs[2]:.6f} zlo zhi\n\n")

        f.write("Masses\n\n")
        f.write("1 12.011\n") 
        if num_types == 2:
            f.write("2 1.008\n")  
        f.write("\n")

        f.write("Atoms # full\n\n")
        for i, (t, (x, y, z)) in enumerate(zip(atom_types, pos), start=1):
            f.write(f"{i} 1 {t} 0.0 {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"[INFO] LAMMPS data written to {filename}")

# =====================================================================
# --- MAIN EXECUTION BLOCK ---
# =====================================================================
if __name__ == "__main__":
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
        
    output_dir = os.path.join(os.path.dirname(base_dir), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"graphene_{STRUCTURE_TYPE}_initial.data")

    print(f"[INFO] Generating {STRUCTURE_TYPE} structure...")
    base_sheet = create_base_graphene()

    if STRUCTURE_TYPE == "flat":
        carbon_scaffold = base_sheet
    elif STRUCTURE_TYPE == "pillars":
        stack = create_graphite_stack(base_sheet, PILLAR_HEIGHT, INTERLAYER_DIST, C_C_BOND)
        carbon_scaffold = carve_pillars(stack, PILLAR_RADIUS, PILLAR_PERIOD)
    else:
        raise ValueError("Invalid STRUCTURE_TYPE. Use 'flat' or 'pillars'.")

    print("[INFO] Passivating edges...")
    passivated_structure = passivate_edges(carbon_scaffold, ch=C_H, min_sep=MIN_SEP, inplane=True)

    geo_data = make_geo(passivated_structure)
    if check_overlaps(geo_data, cutoff=0.90):
        print(" Close atomic contacts detected. Tune MIN_SEP or geometry.")

    write_lammps_full(passivated_structure, output_file)

    positions = passivated_structure.get_positions()
    mins = np.min(positions, axis=0)
    maxs = np.max(positions, axis=0)
    dims = maxs - mins
    
    print("\n--- Graphene Structure Dimensions ---")
    print(f"X-range: {mins[0]:.4f} to {maxs[0]:.4f} A")
    print(f"Y-range: {mins[1]:.4f} to {maxs[1]:.4f} A")
    print(f"Z-range: {mins[2]:.4f} to {maxs[2]:.4f} A")
    print(f"Dimensions: {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f} A")
    print("-------------------------------------")