# S5P-LNO2-Notebook

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7549879.svg)](https://doi.org/10.5281/zenodo.7549879)

Some notebooks for [S5P-LNO2](https://github.com/zxdawn/S5P-LNO2).

## Figures and tables in peer-reviewed papers

### ES&T paper

Spaceborne observations of lightning NO2 in the Arctic (In Review)

Input data are all saved in the zenodo Dataset called [Dataset for "Spaceborne observations of lightning NO2 in the Arctic"]([10.5281/zenodo.7528871](https://doi.org/10.5281/zenodo.7528871)). [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7528872.svg)](https://doi.org/10.5281/zenodo.7528872)

Users can download the compressed file, extract it in the root directory, and rename to `data`. Then, all the Jupyter Notebooks should work well.

- workflow.ipynb
    - Overview of the process for selecting lightning NO2 pixels from TROPOMI (Fig. 1 and Fig. S4)
- lightning_distribution.ipynb
    - Lightning distribution recorded by OTD and GLD360 (Fig. 2 and S2)
    - Time series of Arctic lightning rates (Fig. S7)
    - Time series of Arctic CAPE and AOD (Fig. S8)
    - GLD360 stroke counts, LNO2 emissions, and mean CAPE (Table S1)
- nox_emission.ipynb
    - Lightning and anthropogenic NO2 column densities (Fig. 3)
    - Lightning and anthropogenic NOx emissions (Fig. 6)
- lightning_no2_production.ipynb
    - Distribution of lightning and lightning NO2 productions (Fig. 4)
    - Relationship between production and stroke rate over different regions (Fig. 5)
    - Top 10 lightning NO2 production efficiencies (Table S3)
- lnox_profile_fit.ipynb
    - Fit Gaussian distributions to lightning NOx profiles (Fig. S1)
- large_lnox.ipynb
    - Case of the largest lightning NO2 column (Fig. S5)
- cloud_pressure.ipynb
  - Histogram of TROPOMI cloud pressure and tropopause pressure (Fig. S3)
- no2_above_cloud.ipynb
  - Histograms of TROPOMI cloud pressure and above-cloud NO2 column (Fig. S6)
- no2_lifetime.ipynb
    - Lightning NO2 lifetime estimation (Table S2)
- pe_uncertainty.ipynb
    - Uncertainty of estimated lightning NO2 production efficiency (Table S4)
## Other useful notebooks

- TM5_LNO2.ipynb
  - Check whether lightning NO2 is included in TM5 a priori profile
- bAMF_pressure.ipynb
  - Detailed of calculation of box-AMF

- cluster_stroke_to_flash.ipynb
  - Cluster lightning strokes to flashes
- lightning_in_swaths.ipynb
  - Method of checking the number of lightning points within TROPOMI swath
