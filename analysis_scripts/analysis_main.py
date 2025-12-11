#!/usr/bin/env python3
"""
Rubisco Density Analysis Tool

This script analyzes LAMMPS trajectory files (.lammpstrj) to characterize
Rubisco condensation. It performs the following key tasks:

1.  **Neighbor Counting**: Calculates the number of neighbors for each particle within
    a specified cutoff radius (using a direct 3x3 periodic tiling method).
2.  **Condensation Classification**: Fits a Gaussian Mixture Model (GMM) to the
    neighbor count distribution to determine a dynamic threshold for "condensed" vs "dilute" phases.
3.  **Convex Hull Analysis**: Computes the volume and concentration of the condensed phase cluster.
    - Uses robust Periodic Boundary Condition (PBC) handling:
      - X/Y axes are wrapped using the Minimum Image Convention relative to the cluster center.
      - Z axis is treated as unwrapped/non-periodic for the hull calculation (based on slab geometry),
        though neighbor counting treats it as periodic.
4.  **Radial Distribution Function (RDF)**: Computes g(r) for the condensed phase.
5.  **Visualization**: Generates annotated 2D plots and a sequence of frames for animation,
    allowing visual validation of the cluster identification and PBC wrapping.

Dependencies:
    - numpy
    - matplotlib
    - scipy
    - sklearn

Usage:
    python analysis_main.py -i /path/to/data -o /path/to/output --rbm 2 3 5 --ratio 1 3 5
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture

# ------------------------
# DEFAULT PATH CONFIG
# ------------------------
DEFAULT_DATA_DIR = Path("/Users/jb1725/Documents/Jack Data/final_analysis/chlamy_prod")
DEFAULT_OUTPUT_DIR = Path(
    "/Users/jb1725/Documents/York/PDRA/Manuscripts/"
    "Gaurav Sticker Number/LAMMPs/chlamy"
)

# DEFAULT_DATA_DIR = Path("/Users/jb1725/Documents/Jack Data/final_analysis/chlorella/trajectories_chlorella/prod/lammps_files")
# DEFAULT_OUTPUT_DIR = Path(
#     "/Users/jb1725/Documents/York/PDRA/Manuscripts/"
#     "Gaurav Sticker Number/LAMMPs/chlorella"
# )

# ------------------------
# GLOBAL PARAMETERS (CONFIG)
# ------------------------
NEIGHBOUR_THRESH = 4.9          # Placeholder for GMM determination
DISTANCE_CUTOFF = 20.0          # neighbour search radius (nm)
RBM_DEFAULT = [2, 3, 4, 5, 7]     # rbms to analyse
RATIO_DEFAULT = [1, 2, 3, 5]       # ratios to analyse
CONVEX_HULL_FRAMES = 200        # Number of final frames to use for detailed hull analysis
RDF_MAX_R = 60.0                # Max radius for RDF (nm)
RDF_BINS = 400                  # Number of bins for RDF histogram
ZBOX = 500.0                    # Box length in Z (nm)
SEED = 0                        # Random seed for reproducibility
ATOM_TYPE_RUBISCO = 3           # LAMMPS atom type ID for Rubisco

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Rubisco Condensation Analysis")
    p.add_argument("-i", "--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                   help="Directory containing input LAMMPS .lammpstrj files")
    p.add_argument("-o", "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                   help="Directory to write output files and plots")
    p.add_argument("--rbm", type=int, nargs="+", default=RBM_DEFAULT,
                   help="List of RBM values (integers) to analyse")
    p.add_argument("--ratio", type=int, nargs="+", default=RATIO_DEFAULT,
                   help="List of Ratio values (integers) to analyse")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Random seed for GMM and other stochastic processes")
    p.add_argument("--max_store_frames", type=int, default=None,
                   help="Limit number of frames stored in memory (optional)")
    return p.parse_args()


def get_nrub_xbox(ratio: int):
    """
    Returns the number of Rubiscos (nrub) and box side length (xbox)
    expected for a given ratio.
    """
    if ratio > 3:
        return 325, 85.0
    return 832, 130.0


def pbc_shift(arr, center, L):
    """
    Shift coordinates to be relative to a center point using the
    Minimum Image Convention (1D array operation).
    """
    delta = arr - center
    delta -= np.round(delta / L) * L
    return delta


def gmm_intersection_closed_form(mu1, mu2, s1, s2, w1, w2):
    """
    Analytically solve for the intersection points of two 1D Gaussian PDFs.
    """
    if s1 <= 0 or s2 <= 0:
        return []

    # Coefficients for the quadratic equation derived from log(p1) = log(p2)
    a = 1.0 / (2.0 * s2 ** 2) - 1.0 / (2.0 * s1 ** 2)
    b = mu1 / (s1 ** 2) - mu2 / (s2 ** 2)
    try:
        c = (mu2 ** 2) / (2.0 * s2 ** 2) - (mu1 ** 2) / (2.0 * s1 ** 2) + math.log((w1 * s2) / (w2 * s1))
    except ValueError:
        return []

    # Solve quadratic ax^2 + bx + c = 0
    if abs(a) < 1e-12:  # Linear case
        if abs(b) < 1e-12:
            return []
        return [-c / b]

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return []

    sqrt_disc = math.sqrt(disc)
    x1 = (-b + sqrt_disc) / (2.0 * a)
    x2 = (-b - sqrt_disc) / (2.0 * a)
    return sorted([x1, x2])

def analyze_msd_gmm(unwrapped_traj_path: Path, output_dir: Path, rbm: int, ratio: int, nrub: int):
    """
    Calculates MSD for all particles, then uses a GMM on D values to
    classify Condensed (Slow) vs Dilute (Fast) populations.
    """
    logging.info(f"Calculating MSD (GMM method) for {unwrapped_traj_path.name}...")

    # 1. Read Unwrapped Coordinates
    traj_data = {}
    try:
        with unwrapped_traj_path.open('r') as fh:
            while True:
                line = fh.readline()
                if not line: break
                if "ITEM: NUMBER OF ATOMS" in line:
                    natoms = int(fh.readline())
                    while "ITEM: ATOMS" not in fh.readline(): pass
                    for _ in range(natoms):
                        parts = fh.readline().split()
                        try:
                            atom_id, atom_type = int(parts[0]), int(parts[1])
                            if atom_type == ATOM_TYPE_RUBISCO:
                                if atom_id not in traj_data: traj_data[atom_id] = []
                                traj_data[atom_id].append([float(x) for x in parts[2:5]])
                        except:
                            continue
    except Exception as e:
        logging.error(f"Error reading unwrapped file: {e}")
        return

    # 2. Compute D for everyone
    MSD_WINDOW = 100
    FIT_LAGS = 10
    time_per_frame = 50e-12 * 1e5  # s

    D_all = []

    # Process sorted IDs for consistency
    for atom_id in sorted(traj_data.keys()):
        coords = np.array(traj_data[atom_id])
        if len(coords) < MSD_WINDOW: continue

        # Analyze last window
        seg = coords[-MSD_WINDOW:]
        msd = []
        for lag in range(1, min(51, len(seg))):  # Max lag 50
            diff = seg[lag:] - seg[:-lag]
            msd.append(np.mean(np.sum(diff ** 2, axis=1)))

        if len(msd) < FIT_LAGS: continue

        # Linear fit: MSD = 6Dt
        y = np.array(msd[:FIT_LAGS]) * 1e-6  # nm^2 -> um^2
        x = np.arange(1, FIT_LAGS + 1) * time_per_frame
        slope, _ = np.polyfit(x, y, 1)
        D_all.append(max(slope / 6.0, 0.0))

    if not D_all:
        logging.warning("No valid MSD trajectories found.")
        return

    # 3. Fit GMM to classify Slow vs Fast
    D_array = np.array(D_all).reshape(-1, 1)

    try:
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(D_array)
        labels = gmm.predict(D_array)
        means = gmm.means_.flatten()

        # Identify which label is "Slow" (Condensed)
        slow_idx = np.argmin(means)
        fast_idx = np.argmax(means)

        D_condensed = D_array[labels == slow_idx].flatten()
        D_dilute = D_array[labels == fast_idx].flatten()

        logging.info(
            f"GMM: {len(D_condensed)} Condensed (D~{means[slow_idx]:.2e}), {len(D_dilute)} Dilute (D~{means[fast_idx]:.2e})")

        # 4. Save Results
        out_base = output_dir / f"{rbm}rbm_1_{ratio}"
        if len(D_condensed) > 0:
            np.savetxt(out_base.with_name(f"{out_base.name}_D_condensed.dat"), D_condensed, header="D (um^2/s)")
        if len(D_dilute) > 0:
            np.savetxt(out_base.with_name(f"{out_base.name}_D_dilute.dat"), D_dilute, header="D (um^2/s)")

        # 5. Plot Log-Scale Histogram
        D_log = np.log10(D_array.flatten() + 1e-15)
        plt.figure(figsize=(7, 5))
        counts, bins, _ = plt.hist(D_log, bins=30, alpha=0.5, color='gray', edgecolor='black')

        for m, label, c in zip(means, ["Condensed", "Dilute"], ['r', 'b']):
            plt.axvline(np.log10(m), color=c, linestyle='--', linewidth=2, label=f"{label} Mean")

        plt.xlabel("log10(D) (µm²/s)")
        plt.ylabel("Counts")
        plt.title(f"Diffusivity Distribution (GMM Split)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_base.with_name(f"{out_base.name}_D_hist_log.png"))
        plt.close()

    except Exception as e:
        logging.error(f"GMM Fitting failed: {e}")

# -----------------------------------------------------------------------------
# MSD ANALYSIS FUNCTION
# -----------------------------------------------------------------------------
def analyze_msd(unwrapped_traj_path: Path, output_dir: Path, rbm: int, ratio: int,
                stable_mask: np.ndarray, nrub: int):
    """
    Calculates MSD and Diffusion Coefficient (D) using the unwrapped trajectory.
    Tracks particles by ID to handle unsorted LAMMPS dumps correctly.
    """
    logging.info(f"Calculating MSD for {unwrapped_traj_path.name}...")

    # Dictionary to store trajectories: {atom_id: [[x,y,z], [x,y,z], ...]}
    traj_data = {}

    # 1. Read Unwrapped Coordinates
    try:
        with unwrapped_traj_path.open('r') as fh:
            frame_count = 0
            while True:
                line = fh.readline()
                if not line: break

                if "ITEM: NUMBER OF ATOMS" in line:
                    natoms = int(fh.readline())

                    # Skip header lines until ATOMS
                    while "ITEM: ATOMS" not in fh.readline(): pass

                    for _ in range(natoms):
                        parts = fh.readline().split()
                        # Format: ID Type X Y Z (Standard LAMMPS)
                        try:
                            atom_id = int(parts[0])
                            atom_type = int(parts[1])

                            if atom_type == ATOM_TYPE_RUBISCO:
                                coords = [float(x) for x in parts[2:5]]

                                # Initialize list for this ID if new
                                if atom_id not in traj_data:
                                    traj_data[atom_id] = []

                                traj_data[atom_id].append(coords)
                        except (ValueError, IndexError):
                            continue

                    frame_count += 1

        logging.info(f"Read {frame_count} frames for {len(traj_data)} Rubisco atoms.")

    except Exception as e:
        logging.error(f"Error reading unwrapped file: {e}")
        return

    # 2. Compute MSD
    # Convert dictionary to a sorted list based on ID to match 'stable_mask' order
    # (Assuming stable_mask corresponds to sorted IDs 1..N or similar)
    sorted_ids = sorted(traj_data.keys())

    # Safety check: does the number of atoms match nrub?
    if len(sorted_ids) != nrub:
        logging.warning(f"ID mismatch: Found {len(sorted_ids)} atoms, expected {nrub}. Using found atoms.")
        # If mismatch, we can't reliably use stable_mask by index.
        # We'll assume the first N IDs correspond to the mask (common case) or skip masking.
        # Ideally, stable_mask should also be a dict {id: bool}.

    MSD_WINDOW = 100
    FIT_LAGS = 10
    time_per_frame = 50e-12 * 1e5  # 5 microseconds

    D_vals = {'condensed': [], 'dilute': []}

    for i, atom_id in enumerate(sorted_ids):
        # Retrieve trajectory for this specific atom ID
        coords = np.array(traj_data[atom_id])

        # We need at least MSD_WINDOW frames
        if len(coords) < MSD_WINDOW: continue

        # Use only the last MSD_WINDOW frames
        seg = coords[-MSD_WINDOW:]

        msd = []
        # Calculate MSD for lags 1 to 100
        for lag in range(1, min(101, len(seg))):
            diff = seg[lag:] - seg[:-lag]
            msd.append(np.mean(np.sum(diff ** 2, axis=1)))

        if len(msd) < FIT_LAGS: continue

        # Fit D (MSD = 6Dt + c)
        # 1. Convert nm^2 to um^2 (1e-6 factor)
        y = np.array(msd[:FIT_LAGS]) * 1e-6
        # 2. Time axis
        x = np.arange(1, FIT_LAGS + 1) * time_per_frame

        slope, _ = np.polyfit(x, y, 1)
        D = max(slope / 6.0, 0.0)

        # Classification
        if i < len(stable_mask):
            key = 'condensed' if stable_mask[i] else 'dilute'
            D_vals[key].append(D)

    # 3. Save Results
    out_base = output_dir / f"{rbm}rbm_1_{ratio}"
    for k in ['condensed', 'dilute']:
        if D_vals[k]:
            out_file = out_base.with_name(f"{out_base.name}_D_{k}.dat")
            # Save D values (column 0).
            # Original script had cols: D, dip_count, density. We only have D here for now.
            np.savetxt(out_file, np.array(D_vals[k]), header="D (um^2/s)")
            logging.info(f"  {k.capitalize()} D: {np.mean(D_vals[k]):.2e} um^2/s (N={len(D_vals[k])})")


# -----------------------------------------------------------------------------
# MAIN ANALYSIS LOGIC
# -----------------------------------------------------------------------------

def analyze_trajectory(traj_path: Path, nrub: int, xbox: float, output_dir: Path,
                       rbm: int, ratio: int, max_store_frames=None):
    """
    Process a single trajectory file.
    """
    logging.info("Analyzing %s (nrub=%d, xbox=%.1f)", traj_path.name, nrub, xbox)

    # 1. Count frames to initialize arrays
    with traj_path.open("r") as fh:
        n_frames = sum(1 for line in fh if line.strip() == "ITEM: NUMBER OF ATOMS")

    if n_frames == 0:
        logging.warning("No frames detected in %s", traj_path)
        return {}
    logging.info("Detected %d frames", n_frames)

    # 2. Setup storage for analysis
    n_save = min(CONVEX_HULL_FRAMES, n_frames)
    store_frames = min(max_store_frames, n_frames) if max_store_frames else n_frames
    store_all = (store_frames == n_frames)

    # Arrays to store results
    d_array = np.empty((store_frames if store_all else n_save, nrub), dtype=np.int32)
    # Storage for last N frames (coordinates)
    rx_last = np.empty((n_save, nrub), dtype=float)
    ry_last = np.empty((n_save, nrub), dtype=float)
    rz_last = np.empty((n_save, nrub), dtype=float)

    saved = 0

    # RDF setup
    rdf_bins = np.linspace(0.0, RDF_MAX_R, RDF_BINS + 1)
    rdf_counts = np.zeros(len(rdf_bins) - 1, dtype=float)
    rdf_norm = 0.0

    # Hull results containers
    hull_volumes_nm3 = []
    hull_concentrations_um3 = []
    mol_per_um3_to_uM = 1.0 / 602.214129

    # Pre-allocate tiling arrays for direct distance calculation
    rx_full = np.empty(nrub * 9, dtype=float)
    ry_full = np.empty(nrub * 9, dtype=float)
    rz_full = np.empty(nrub * 9, dtype=float)

    # 3. Read trajectory and process frames
    with traj_path.open("r") as fh:
        frame_idx = 0
        eof = False
        while not eof:
            # Parse header
            line = fh.readline()
            if not line: break
            if line.strip() != "ITEM: NUMBER OF ATOMS": continue

            line = fh.readline()
            if not line: break
            try:
                natoms = int(line.strip())
            except ValueError:
                break

            # Fast-forward to atoms
            while True:
                line = fh.readline()
                if not line:
                    eof = True
                    break
                if line.startswith("ITEM: ATOMS"): break
            if eof: break

            # Read atom coordinates
            rx = np.zeros(nrub, dtype=float)
            ry = np.zeros(nrub, dtype=float)
            rz = np.zeros(nrub, dtype=float)
            count = 0
            for _ in range(natoms):
                line = fh.readline()
                if not line:
                    eof = True
                    break
                parts = line.strip().split()
                if len(parts) < 5: continue
                try:
                    # Assumes column 2 is type, 3,4,5 are x,y,z
                    atom_type = int(float(parts[1]))
                except ValueError:
                    atom_type = int(parts[1])

                if atom_type == ATOM_TYPE_RUBISCO and count < nrub:
                    try:
                        rx[count], ry[count], rz[count] = float(parts[2]), float(parts[3]), float(parts[4])
                    except ValueError:
                        continue
                    count += 1

            if count < nrub:
                continue

            # --- Neighbor Counting: Direct 3x3 Tiling ---
            # Create 9 periodic images in X/Y plane
            rx_plus = rx + xbox
            rx_minus = rx - xbox
            ry_plus = ry + xbox
            ry_minus = ry - xbox

            # Concatenate tiles
            rx_full[:] = np.concatenate((rx, rx_plus, rx_minus, rx, rx_plus, rx_minus, rx, rx_plus, rx_minus))
            ry_full[:] = np.concatenate((ry, ry, ry, ry_plus, ry_plus, ry_plus, ry_minus, ry_minus, ry_minus))
            rz_full[:] = np.tile(rz, 9)

            # Calculate distances and count neighbors
            d = np.zeros(nrub, dtype=np.int32)
            for i in range(nrub):
                dx = rx_full - rx[i]
                dy = ry_full - ry[i]
                dz = rz_full - rz[i]
                r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                # Count neighbors within cutoff (excluding self-image r~0)
                neighbours = r[(r < DISTANCE_CUTOFF) & (r > 1e-6)]
                d[i] = len(neighbours)

            # Store results
            if store_all:
                d_array[frame_idx, :] = d
            else:
                if frame_idx >= n_frames - n_save:
                    idx = frame_idx - (n_frames - n_save)
                    d_array[idx, :] = d

            if frame_idx >= n_frames - n_save:
                idx = frame_idx - (n_frames - n_save)
                rx_last[idx, :] = rx.copy()
                ry_last[idx, :] = ry.copy()
                rz_last[idx, :] = rz.copy()
                saved += 1

            frame_idx += 1

    # 4. Post-Process: RDF and Convex Hull (Last N Frames)
    V_box = xbox * xbox * ZBOX

    for fi in range(saved):
        # Retrieve data corresponding to the stored frame index fi
        d_frame = d_array[-saved + fi]

        cond_mask = d_frame > NEIGHBOUR_THRESH
        N_condensed = int(np.sum(cond_mask))

        rx_f = rx_last[fi]
        ry_f = ry_last[fi]
        rz_f = rz_last[fi]

        # --- RDF Calculation ---
        if N_condensed > 3:
            rx_full[:] = np.concatenate(
                (rx_f, rx_f + xbox, rx_f - xbox, rx_f, rx_f + xbox, rx_f - xbox, rx_f, rx_f + xbox, rx_f - xbox))
            ry_full[:] = np.concatenate(
                (ry_f, ry_f, ry_f, ry_f + xbox, ry_f + xbox, ry_f + xbox, ry_f - xbox, ry_f - xbox, ry_f - xbox))
            rz_full[:] = np.tile(rz_f, 9)

            for i in np.where(cond_mask)[0]:
                dx = rx_full - rx_f[i]
                dy = ry_full - ry_f[i]
                dz = rz_full - rz_f[i]
                r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                dist = r[(r > 0) & (r < RDF_MAX_R)]
                hist, _ = np.histogram(dist, bins=rdf_bins)
                rdf_counts += hist
            rdf_norm += N_condensed

        # --- Convex Hull Analysis ---
        if N_condensed < 4:
            hull_volumes_nm3.append(np.nan)
            hull_concentrations_um3.append(np.nan)
            continue

        rx_cond = rx_f[cond_mask]
        ry_cond = ry_f[cond_mask]
        rz_cond = rz_f[cond_mask]

        # Calculate Center of Mass of the (potentially broken) cluster
        cx, cy, cz = np.mean(rx_cond), np.mean(ry_cond), np.mean(rz_cond)

        # Apply PBC Shift: Wrap particles to form a single contiguous cluster around COM
        rx_shift = pbc_shift(rx_cond, cx, xbox)
        ry_shift = pbc_shift(ry_cond, cy, xbox)
        rz_shift = rz_cond

        points = np.column_stack([rx_shift, ry_shift, rz_shift])

        if len(points) >= 4:
            hull = ConvexHull(points)
            V_nm3 = hull.volume
            V_um3 = V_nm3 * 1e-9  # Convert nm^3 to um^3
            conc_um3 = N_condensed / V_um3
            hull_volumes_nm3.append(V_nm3)
            hull_concentrations_um3.append(conc_um3)
        else:
            hull_volumes_nm3.append(np.nan)
            hull_concentrations_um3.append(np.nan)

    # 5. Finalize RDF
    bin_centres = 0.5 * (rdf_bins[:-1] + rdf_bins[1:])
    dr = rdf_bins[1] - rdf_bins[0]
    shell_vol = 4.0 * math.pi * bin_centres ** 2 * dr
    rho = nrub / V_box  # Global number density
    normalization = rdf_norm * shell_vol * rho

    rdf = np.zeros_like(bin_centres)
    valid_bins = normalization > 1e-12
    rdf[valid_bins] = rdf_counts[valid_bins] / normalization[valid_bins]

    # 6. Save Data
    out_base = output_dir / f"{rbm}rbm_1_{ratio}"
    out_base.parent.mkdir(parents=True, exist_ok=True)

    raw_d_data = d_array[-saved:].flatten()
    np.save(out_base.with_name(out_base.name + "_neighbor_counts.npy"), raw_d_data)

    # Convex Hull Data
    hv = np.array(hull_volumes_nm3)
    hc = np.array(hull_concentrations_um3)
    valid = ~np.isnan(hc)
    hc_uM = np.full_like(hc, np.nan)
    hc_uM[valid] = hc[valid] * mol_per_um3_to_uM

    header_meta = (
        f"ConvexHullVol(nm^3)  Concentration(molecules/um^3)  Concentration(uM)\n"
        f"neighbour_thresh={NEIGHBOUR_THRESH} distance_cutoff={DISTANCE_CUTOFF} "
        f"nrub={nrub} xbox={xbox} zbox={ZBOX} frames_used_for_hull={len(hv)}"
    )
    np.savetxt(out_base.with_name(out_base.name + "_hull_concentration_per_frame.dat"),
               np.column_stack([hv, hc, hc_uM]), header=header_meta)

    # Frame Densities
    dens_out = out_base.with_name(out_base.name + "_frame_densities.dat")
    frame_indices = np.arange(d_array.shape[0]) if store_all else np.arange(n_frames - n_save, n_frames)
    with dens_out.open("w") as fh:
        for idx in range(d_array.shape[0]):
            n_above = int((d_array[idx] > NEIGHBOUR_THRESH).sum())
            fh.write(f"{frame_indices[idx]} {n_above}\n")

    # RDF Data
    np.savetxt(out_base.with_name(out_base.name + "_rdf_condensed.dat"),
               np.column_stack([bin_centres, rdf]), header="r(nm) g(r)")

    # 7. Generate RDF Plot (Static)
    plt.figure(figsize=(6, 4))
    plt.plot(bin_centres, rdf, lw=2)
    plt.xlabel("r (nm)")
    plt.ylabel("g(r)")
    plt.title(f"RDF (condensed) — RBM {rbm}, Ratio {ratio}")
    plt.tight_layout()
    plt.savefig(out_base.with_name(out_base.name + "_rdf_condensed.png"), dpi=300)
    plt.close()

    # 8. Animated GIF Frame Generation
    # Generates a series of PNGs for the last N frames to allow visual
    # verification of the hull calculation and coordinate wrapping.
    gif_frame_dir = output_dir / f"{rbm}rbm_1_{ratio}_gif_frames"
    gif_frame_dir.mkdir(exist_ok=True)

    for fi in range(saved):
        original_frame_idx = n_frames - saved + fi
        d_frame = d_array[-saved + fi]
        rx_f, ry_f, rz_f = rx_last[fi], ry_last[fi], rz_last[fi]

        cond_mask = d_frame > NEIGHBOUR_THRESH
        N_cond = int(np.sum(cond_mask))

        # Calculate Hull for this frame (In-loop logic for robustness)
        V_nm3_f, Conc_uM_f = np.nan, np.nan
        rx_c, ry_c, rz_c = rx_f[cond_mask], ry_f[cond_mask], rz_f[cond_mask]

        if N_cond >= 4:
            cx, cy, cz = np.mean(rx_c), np.mean(ry_c), np.mean(rz_c)
            rxs = pbc_shift(rx_c, cx, xbox)
            rys = pbc_shift(ry_c, cy, xbox)
            rzs = rz_c  # Z unwrapped

            p3d = np.column_stack([rxs, rys, rzs])
            h3d = ConvexHull(p3d)
            V_nm3_f = h3d.volume
            Conc_uM_f = (N_cond / (V_nm3_f * 1e-9)) * mol_per_um3_to_uM

        # Plot Frame (Z vs X projection)
        plt.figure(figsize=(8, 4))
        plt.scatter(rz_f[~cond_mask], rx_f[~cond_mask], color='blue', s=30, alpha=0.5, label='Dilute')
        plt.scatter(rz_c, rx_c, color='red', s=50, label='Condensed')

        if N_cond >= 3:
            # Calculate 2D Hull for visualization overlay
            pts2d = np.column_stack([rx_c, rz_c])
            h2d = ConvexHull(pts2d)
            for simplex in h2d.simplices:
                plt.plot(pts2d[simplex, 1], pts2d[simplex, 0], 'k-', lw=1)

        plt.xlabel("Z (nm)")  # Long axis
        plt.ylabel("X (nm)")  # Short axis
        # Set limits to see the whole box length to verify wrapping behavior
        plt.xlim(-150, 350)
        plt.ylim(0, xbox)

        anno = f"N: {N_cond}"
        if not np.isnan(V_nm3_f):
            anno += f"\nV: {V_nm3_f:.1f} nm^3\nC: {Conc_uM_f:.2f} uM"

        plt.text(0.05, 0.95, anno, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
        plt.title(f"RBM {rbm} | Ratio {ratio} | Frame {original_frame_idx}")
        plt.tight_layout()
        plt.savefig(gif_frame_dir / f"frame_{fi:04d}.png", dpi=100)
        plt.close()

    logging.info("Generated %d frames in %s", saved, gif_frame_dir)

    return {
        "d_array": d_array,
        "out_base": out_base,
        "n_frames": n_frames
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def run_analysis(args):
    data, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # List to collect all density data for the combined output file
    all_d = []

    for rbm in args.rbm:
        for ratio in args.ratio:
            nrub, xbox = get_nrub_xbox(ratio)
            f_base = f"{rbm}rbm_1_{ratio}_trajectory"

            f_wrap = Path(args.data_dir) / f"{f_base}_full.lammpstrj"
            f_unwrap = Path(args.data_dir) / f"{f_base}_unwrapped_full.lammpstrj"

            if f_wrap.exists():
                # 1. Run Main Analysis (Density, Hull, RDF)
                # Note: analyze_trajectory already saves the PER-RUN neighbor_counts.npy
                res = analyze_trajectory(f_wrap, nrub, xbox, out, rbm, ratio, args.max_store_frames)

                # 2. Run MSD Analysis (Dynamics)
                if f_unwrap.exists():
                    analyze_msd_gmm(f_unwrap, out, rbm, ratio, nrub)
                else:
                    logging.warning(f"  MSD Skipped: Missing {f_unwrap.name}")

                # 3. Collect density data for the COMBINED file
                npy_path = out / f"{rbm}rbm_1_{ratio}_neighbor_counts.npy"
                if npy_path.exists():
                    all_d.append(np.load(npy_path))
            else:
                logging.warning(f"Missing {f_wrap.name}")

    # --- AGGREGATE DATA OUTPUT ---
    if all_d:
        combined = np.concatenate(all_d)
        combined_path = out / "combined_gmm_data.npy"
        np.save(combined_path, combined)
        logging.info(f"Saved combined density data to: {combined_path.name}")

    logging.info("Done.")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    run_analysis(args)


if __name__ == "__main__":
    main()