# linear-eigenvalue-problems-in-photonic-crystals
A kernel compensation method and eigendecomposition of double curl operator.

This repository provides **MATLAB** and **Python** implementations for computing frequencies, bandgaps of photonic crystals. The project applies locally optimal block preconditioned conjugate gradient (**lobpcg**) methods to solve problems in photonic crystals (PCs) and includes several modules to handle dielectric properties, discretization, and eigenvalue solving.

## Project Structure
### MATLAB Directory: PCs_mfdm_lobpcg_matlab
This directory contains MATLAB scripts and functions organized into specific folders for modular functionality.

dielectric/: Contains functions related to dielectric properties.

discretization/: Contains functions for the spatial discretization of the computational domain.

lobpcg/: Implementation of the lobpcg method for eigenvalue problems.

output/: Scripts for plotting and running experiments.

plot_bandgap.m: Plots the bandgap based on computed data.

plot_runtime_table.m: Generates runtime comparison tables.

run_1p.m: Computes frequencies of a single lattice vector.

run_gapplot.m: Executes the gap plotting script.

run_scalcmp.m: Executes scaling comparison tests.

run_timecmp.m: Executes time comparison tests.

run_tolcmp.m: Executes tolerance comparison tests.

### Python Directory: PCs_mfdm_lobpcg_python
This directory provides the Python version of the project, with modular components and script files.

__pycache__/: Stores cached Python files.

diel_info.json: JSON file containing dielectric information.

dielectric.py: Handles dielectric properties.

discretization.py: Implements spatial discretization for the problem domain.

eigen_solver.py: Implements the eigenvalue solver using the lobpcg method.

gpu_opts.py: Norms (package), timing (gpu timing requires synchronization).

output.py: Manages output data and results visualization.

run.py: Main script to execute the Python-based simulations.

## How to implement functions
### Matlab

### Python
