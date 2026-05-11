# Changelog

All notable changes to this repository are documented here.
Versions correspond to arXiv submissions of the paper.

---

## [v0.3.0-arxiv_v3] - 2026-05-11
### Added
- Test suite to test main simulation scripts in `tests/`
- `scripts/symm_net/scatter_stdwi_rdd_sal.ipynb` to reproduce paper figure no. 7
- Added code for convenient reproduction
 of figure 8 and figure 9 end-to-end (i.e. `scripts/symm_net/sweep*.py*`, `scripts/symm_net/plots.ipynb` and `scripts/symm_net/plot_puresymm.ipynb`)

### Updated
- Update all docstrings in a consistent form if needed.
- Add consistent typing where needed.
- README.md to match current repo structure.
- All plotting scripts if figures have changed.

### Remove
- obsolete code chunks in `spiking_sampling_network/src/neuralsampling`
- obsolete code chunks in `spiking_microcircuits/src/microcircuits`


## [v0.2.0-arxiv_v2] - 2026-03-24
### Added
- Symmnet experiments for figures 7 and 8 (`scripts/symmnet/`) plus `symmnet` package
- Dale's law experiments for figure 9 (`scripts/dales_law/`) plus the addon for the `neuralsampling` package (`spiking_sampling_network/src/neuralsampling/eisystem.py`)

---

## [v0.1.0-arxiv_v1] - 2025-03-13
### Added
- Initial release accompanying arXiv v1
- SSN experiments (figures 4 & 5, `spiking_sampling_network` and `scripts/ssn/`)
- Microcircuit experiments (figure 6, `spiking_microcircuits` and `scripts/microcircuits/`)
- Code for analytically calculating the STDDs (`stdd_calculator`)
- Comparison of different psp shapes (figure 7, `scripts/psp_shapes.ipynb`)
