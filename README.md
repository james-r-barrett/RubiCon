# RubiCon: Rubisco Condensation Analysis Toolkit

RubiCon (Rubisco Condensation) was designed to analyze and quantify phase separation in LAMMPS molecular dynamics simulations of Rubisco and linker proteins.  
It provides a robust pipeline for:

- Detecting condensates  
- Calculating phase-specific properties (concentration, volume)  
- Analyzing dynamics (diffusion)

## Features

### Condensate thresholding
- Uses neighbor counts to identify dilute vs. condensed phases.

### Geometric Analysis
- Calculates the volume and concentration of condensed clusters using 3D Convex Hulls with periodic boundary condition (PBC) handling.

### Dynamics (MSD)
- Computes Mean Squared Displacement (MSD) and diffusion coefficients (D).
- Uses Gaussian Mixture Models (GMM) to classify particles as “fast” (dilute) or “slow” (condensed).

### Structural Analysis
- Computes Radial Distribution Functions (RDF) for the condensed phase.

### Publication-Ready Plotting
- Generates PDFs including Raincloud plots, boxplots, and cluster statistics histograms.

## Dependencies

Requires Python 3.8+:

```
pip install numpy matplotlib scipy scikit-learn pandas ptitprince
```

- numpy & scipy: numerical + spatial analysis  
- scikit-learn: GMM clustering  
- matplotlib: plotting  
- pandas & ptitprince: raincloud plots  

## Repository Structure

```
RubiCon/
├── analysis_main.py
├── plot.py
├── animate_gif.py/
└── README.md
```

## 1. Running the Analysis

Input files must follow naming pattern:

```
{RBM}rbm_1_{Ratio}_trajectory_full.lammpstrj
```

### Usage

```
python rubisco_analysis.py     -i /path/to/lammps/data     -o /path/to/output/results     --rbm 2 3 5 7 9     --ratio 1 2 3 5     --max_store_frames 2000
```

### Outputs

- *_neighbor_counts.npy  
- *_hull_concentration_per_frame.dat  
- *_frame_densities_JB.dat  
- *_rdf_condensed.dat  
- *_D_condensed.dat / *_D_dilute.dat  

## 2. Visualizing Results

```
python plot_rubisco_data.py -o /path/to/output/results
```

### Generated Figures

- neighbor_count_histogram.pdf  
- chlamy_fraction_condensed_iqr.pdf  
- convex_hull_concentration...pdf  
- rdf_analysis_stats.pdf  
- condensed_raincloud.pdf  

## Methodology Details

### Phase Classification
Uses KDE minima of neighbor count distributions to define condensed phase.

### Convex Hull & PBC
Largest cluster identified → wrapped coordinates → 3D convex hull volume.

### Diffusivity (MSD)
MSD fits → diffusion coefficient D → GMM on log(D) to classify phases.

## License

MIT License.
