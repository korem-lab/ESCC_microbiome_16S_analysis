# ESCC_microbiome_16S_analysis

This repository contains the results and code used to generate figures supporting our manuscript titled "A generalizable cross-continent prediction of esophageal squamous cell carcinoma using the oral microbiome"

| Folder/file              | Description                                                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.py`               | script to generate all figures                                                                                                                                                              |
| `data/`                | contains metadata, ASV tables, and taxonomies assigned to ASVs                                                                                                                              |
| `code/`                | contains helper functions                                                                                                                                                                   |
| `results/`             | contains figures and tables                                                                                                                                                                 |
| `results/figures/`     | contains all figure images                                                                                                                                                                  |
| `results/diversity/`   | contains weighted and unweighted UniFrac distance matrices                                                                                                                                  |
| `results/tables/`      | contains table one statistics and supplementary tables                                                                                                                                      |
| `results/corncob/`     | contains results from corncob (https://github.com/statdivlab/corncob) differential abundance analyses,<br />including models adjusted for batch only and models adjusted for all covariates |
| `results/predictions/` | contains predicted values from all models, within and cross-study. also contains SHAP values from<br />model evaluation                                                                     |
