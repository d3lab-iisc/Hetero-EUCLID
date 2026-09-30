# Hetero-EUCLID

**Interpretable model discovery for heterogeneous hyperelastic materials using stress-unsupervised learning**

[![Paper](https://img.shields.io/badge/CMAME-10.1016%2Fj.cma.2026.118729-blue)](https://doi.org/10.1016/j.cma.2026.118729)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)

This repository contains the code and example datasets for the paper:

> K. L. Chaurasiya, S. Dutta, S. Kumar, A. Joshi.
> **Hetero-EUCLID: Interpretable model discovery for heterogeneous hyperelastic materials using stress-unsupervised learning.**
> *Computer Methods in Applied Mechanics and Engineering*, 452, 118729 (2026).
> https://doi.org/10.1016/j.cma.2026.118729

Developed at the [Dynamics, Data and Design (D3) Lab](https://d3lab-iisc.github.io/), Department of Mechanical Engineering, Indian Institute of Science (IISc) Bengaluru.

## Abstract

We propose a computational framework, Hetero-EUCLID, for segmentation and parameter identification to characterize the full hyperelastic behavior of all constituents of a heterogeneous material. In this work, we leverage the Bayesian-EUCLID (Efficient Unsupervised Constitutive Law Identification and Discovery) framework to efficiently solve the heterogenized formulation through parsimonious model selection using sparsity-promoting priors and Monte Carlo Markov Chain sampling. We utilize experimentally observable 3D surface displacement and boundary-averaged force data generated from Finite Element simulations of non-equi-biaxial tension tests on heterogeneous specimens. The framework broadly consists of two steps– residual force-based segmentation and constitutive parameter identification. We validate and demonstrate the ability of the proposed framework to segment the domain and characterize the constituent materials on various types of thin square heterogeneous domains. We validate the framework’s ability to segment and characterize materials with multiple levels of displacement noises and non-native mesh discretizations, i.e, using different meshes for the forward FE simulations and the inverse EUCLID problem. This demonstrates the applicability of the Hetero-EUCLID framework in Digital Image/Volume Correlation-based experimental scenarios. Furthermore, the proposed framework performs successful segmentation and material characterizations based on data from a single experiment, thereby making it viable for rapid, interpretable model discovery in domains such as aerospace and defense composites and for characterization of selective tissue stiffening in medical conditions such as fibroatheroma, atherosclerosis, or cancer.

## Relation to Bayesian-EUCLID

Hetero-EUCLID extends [Bayesian-EUCLID](https://github.com/EUCLID-code/EUCLID-hyperelasticity-bayesian) (Joshi et al., *CMAME* 2022) to heterogeneous domains. The hierarchical Bayesian spike-and-slab solver, the Gibbs sampler and the kinematics utilities in `core/` build on the Bayesian-EUCLID implementation. For a detailed description of those modules and functions, see the [Bayesian-EUCLID documentation](https://euclid-code.github.io/EUCLID-hyperelasticity-bayesian/mkdocs/site/).

The pipeline in `drivers/main_hss_hetero_euclid.py` runs these steps:

1. **Assemble the system**: load the FE displacement and reaction-force data (6-node wedge elements, 2 integration points) and build the force-balance equations for a single homogeneous material.
2. **Segment using residual forces**: fit the homogeneous model by least squares, compute the nodal residual forces, and grow material "islands" from regions where the residuals are high (`growIslands`).
3. **Assemble the heterogeneous system**: rebuild the equations with one set of material parameters per segmented island.
4. **Identify parameters (Bayesian)**: run parallel spike-and-slab MCMC chains to find a sparse, interpretable constitutive model for each constituent, with uncertainty.

If you add noise, the displacements are perturbed with Gaussian noise and then denoised using kernel ridge regression before step 1.

## Repository structure

```
Hetero-EUCLID/
├── core/                                   # Solver modules (adapted from Bayesian-EUCLID)
│   ├── core_spike_slab.py                  # Hierarchical Bayesian spike-and-slab Gibbs sampler
│   ├── truncated_multivariate_normal.py    # Minimax-tilting TMVN sampler (Botev 2016; impl. P. Brunzema)
│   ├── unsupervised_hyperelasticity.py     # Data containers, filtering and processing of A, b
│   ├── load_fem_data_wedge_2intpoints.py   # FE data loader (wedge elements), noise + KRR denoising
│   ├── data_definitions.py, utilities.py   # Kinematics, invariants, reference state
│   ├── helper.py, post_process.py          # Misc. helpers and plotting
├── drivers/
│   ├── main_hss_hetero_euclid.py           # Main entry point
│   ├── config.py                           # Hyperparameters (priors, chain length, seed, ...)
│   ├── features_library.py                 # Constitutive feature library
│   ├── preprocess_data_hetero.py           # Assembles the linear system per material cluster
│   ├── growth_paraview_2_residualForces.py # Residual-force based segmentation (island growth)
│   └── paraview_plot.py                    # Writes .vtk output for ParaView
├── main_data/
│   ├── Type1_Cross_Shaped_Inclusion/{1..6}/  # FE data per load step
│   └── Type2_Split_Domain/{1..6}/
├── HeteroEUCLID.yml                        # Exact conda environment (Windows)
└── requirements.txt                        # Cross-platform pip requirements
```

Each load-step folder contains the FE data used as input:

| File | Content |
|---|---|
| `output_nodes.csv` | Node id, reference coordinates `x,y,z`, displacements `ux,uy,uz`, Dirichlet BC flags `bcx,bcy,bcz`, `mat_id` |
| `output_elements.csv` | 6-node wedge connectivity `node1..node6`, ground-truth `mat_id` |
| `output_reactions.csv` | Boundary-averaged reaction forces |

## Installation

The code was developed and tested with Python 3.12 on Windows. Clone the repository first:

```bash
git clone https://github.com/d3lab-iisc/Hetero-EUCLID.git
cd Hetero-EUCLID
```

### Option A: conda (exact environment, Windows)

`HeteroEUCLID.yml` pins the exact package builds used for the paper. It is Windows-specific.

```bash
conda env create -f HeteroEUCLID.yml
conda activate HeteroEUCLID
```

### Option B: pip (any OS)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

The code runs on the CPU. On Linux, you can install the smaller CPU-only PyTorch wheel first with `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu`, then run `pip install -r requirements.txt`.

## Usage

The scripts use paths relative to `drivers/`, so run them from that folder:

```bash
cd drivers
python main_hss_hetero_euclid.py <case> <noise>
```

| Argument | Values |
|---|---|
| `<case>` | `Type1_Cross_Shaped_Inclusion`, `Type2_Split_Domain` (any folder name inside `main_data/`) |
| `<noise>` | `none` (0), `low` (1e-3), `moderate` (2e-3), `high` (5e-3): the standard deviation of the Gaussian displacement noise |

Examples:

```bash
python main_hss_hetero_euclid.py Type1_Cross_Shaped_Inclusion none
python main_hss_hetero_euclid.py Type2_Split_Domain moderate
```

Hyperparameters such as the spike-and-slab priors (`a_v, b_v, a_p, b_p, a_sigma, b_sigma`), the chain length, the burn-in, the number of parallel chains and the random seed are set in [`drivers/config.py`](drivers/config.py).

### Noisy cases

For `low`, `moderate` or `high`, the script first copies the case to `main_data/<case>_denoised_<level>/` and works on that copy, so the original data is not changed. The denoised displacements overwrite `output_nodes.csv` in the copy. **Delete that folder before running the same noise level again**, otherwise the noise is added a second time to data that has already been denoised.

## Output

The results are written to the analysed load-step folder (`main_data/<case>/6/`):

| File | Content |
|---|---|
| `theta_stats.csv` | Posterior mean (`theta_j`) and standard deviation (`std_j`) of the material parameters. There is one row per segmented material island. The last column is the wall-clock time |
| `theta_ols.csv` | Least-squares estimate of the parameters per island (for reference) |
| `chain.pkl` | Combined MCMC chains (pickled `Chain` object) |
| `output_elements.csv` | Updated in place with the segmentation result (`island_id` column) |
| `residuals_*.csv`, `thresholding_res_mask.csv` | Residual forces and the mask used for segmentation |
| `step_6_output.vtk` | Mesh with the segmentation and residual fields, for viewing in ParaView |
| `A1.csv, b1.csv, A2.csv, b2.csv, features.csv, ...` | Intermediate assembled system (large files, regenerated on every run) |

The parameter columns `theta_0 … theta_5` correspond to the feature library in `drivers/features_library.py`:

| Index | Feature |
|---|---|
| 0 | $(\tilde I_1 - 3)$ |
| 1 | $(\tilde I_2 - 3)$ |
| 2 | $(\tilde I_1 - 3)^2$ |
| 3 | $(\tilde I_1 - 3)(\tilde I_2 - 3)$ |
| 4 | $(\tilde I_1 - 3)^3$ |
| 5 | $(J - 1)^2$ |

Here $\tilde I_1 = I_1 J^{-2/3}$ and $\tilde I_2 = I_2 J^{-4/3}$ are the isochoric invariants of the right Cauchy–Green tensor.

## Citation

If you use this code, please cite:

```bibtex
@article{chaurasiya2026heteroeuclid,
  title   = {Hetero-EUCLID: Interpretable model discovery for heterogeneous hyperelastic materials using stress-unsupervised learning},
  author  = {Chaurasiya, Kanhaiya Lal and Dutta, Saurav and Kumar, Siddhant and Joshi, Akshay},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {452},
  pages   = {118729},
  year    = {2026},
  doi     = {10.1016/j.cma.2026.118729}
}
```

and the Bayesian-EUCLID framework it builds on:

```bibtex
@article{joshi2022bayesian,
  title   = {Bayesian-EUCLID: Discovering hyperelastic material laws with uncertainties},
  author  = {Joshi, Akshay and Thakolkaran, Prakash and Zheng, Yiwen and Escande, Maxime and Flaschel, Moritz and De Lorenzis, Laura and Kumar, Siddhant},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {398},
  pages   = {115225},
  year    = {2022},
  doi     = {10.1016/j.cma.2022.115225}
}
```

## License

This project is released under the [GNU General Public License v3.0](LICENSE), the same license as Bayesian-EUCLID, on which it is based.

## Contact

This code is maintained by the [D3 Lab](https://d3lab-iisc.github.io/) at IISc Bengaluru.