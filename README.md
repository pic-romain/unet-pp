# Distributional Regression U-Nets for the Postprocessing of Precipitation Ensemble Forecasts

This repository contains the code for the article "Distributional Regression U-Nets for the Postprocessing of Precipitation Ensemble Forecasts" by R. Pic, C. Dombry, P. Naveau, and M. Taillardat available on [arXiv](https://arxiv.org/abs/2407.02125) and [HAL](https://hal.archives-ouvertes.fr/hal-04631942).

```bibtex	
@misc{Pic2024DRU,
    title={Distributional Regression U-Nets for the Postprocessing of Precipitation Ensemble Forecasts},
    author={Romain Pic and Clément Dombry and Philippe Naveau and Maxime Taillardat},
    year={2024},
    eprint={2407.02125},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2407.02125}, 
}
```

This repository does not provide the data required to run the code. The data used in the article is not available. The code is provided as is and can be used to reproduce the results of the article on the data of the user. Guidance is provided on the expected data shapes and the description of the files.

Table of contents
---
  - [Data](#data)
    - [Expected data shapes](#expected-data-shapes)
  - [Description of the files](#description-of-the-files)
    - [References models](#references-models)
    - [U-Net-based methods](#u-net-based-methods)
    - [Metrics](#metrics)
  - [References and dependencies](#references-and-dependencies)
  - [Feedback](#feedback)


## Data

**The data of the article is not available.**

The code of this repository relies on data in the form of numpy arrays in the `data` folder. The `data` folder should contain the following files:
- `X_trainval.npy`, `Y_trainval.npy` : predictors and observations for the training/validation set.
- `X_test.npy`, `Y_test.npy` : predictors and observations for the test set.
- `trainval_dow.npy` : day of the week for the data in the training/validation set.
- `X_constant.npy` : constant fields (e.g., orography) used for both the training/validation and test sets. 
- `X_raw_trainval.npy`, `X_raw_test.npy` : raw ensemble forecasts for the training/validation and test sets.

### Expected data shapes

| File name            | Expected shape                           |
| -------------------- | ---------------------------------------- |
| `X_trainval.npy`     | ($n_{trainval}$, $H$, $W$, $n_{pred}$)   |
| `X_test.npy`         | ($n_{test}$, $H$, $W$, $n_{pred}$)       |
| `Y_trainval.npy`     | ($n_{trainval}$, $H$, $W$)               |
| `Y_test.npy`         | ($n_{test}$, $H$, $W$)                   |
| `trainval_dow.npy`   | ($n_{trainval}$,)                        |
| `X_constant.npy`     | ($H$, $W$, $n_{constant}$)               |
| `X_raw_trainval.npy` | ($n_{trainval}$, $H$, $W$, $n_{member}$) |
| `X_raw_test.npy`     | ($n_{test}$, $H$, $W$, $n_{member}$)     |

where:
- $n_{trainval}$ is the number of samples in the training/validation set,
- $n_{test}$ is the number of samples in the test set,
- $W$ and $H$ are the width and height of the grid considered,
- $n_{pred}$ is the number of predictors (without the constant fields),
- $n_{constant}$ is the number of constant fields,
- $n_{member}$ is the number of members in the raw ensemble.

## Description of the files

The following sections describe the files of the repository in the order of a standard workflow. The `utils` folder contains utility functions used in the scripts. The `output` folder contains the results of the scripts both raw results and figures.

### References models

The models are trained using the predictors (`X_trainval.npy`) and the observations (`Y_trainval.npy`). The models are saved in the `output/reference_models/models` folder. Hyperparameters can be provided as arguments.

[**Quantile Regression Forests**](https://doi.org/10.1175/MWR-D-15-0260.1) **(QRF)**
- `qrf_pred.R` : R script to train a QRF at each grid point using the predictors. 

[**QRF with tail extension**](https://doi.org/10.1175/WAF-D-18-0149.1) **(TQRF)**
- `qrf+gtcnd_pred.R` : R script to train a TQRF for a generalized truncated/censored normal distribution (GTCND), at each grid point using the predictors.
- `qrf+csgd_pred.R` : R script to train a TQRF for a censored shifted gamma distribution (CSGD), at each grid point using the predictors.

### U-Net-based methods

The U-Net-based methods are trained on the predictors (`X_trainval.npy`)  and the observations (`Y_trainval.npy`). The models are saved in the `output/unet_models/parameters` folder. Hyperparameters can be provided as arguments.

- `unet_pred.py` : Python script to train a U-Net model over the whole grid.
- `group_seq.py` : Python script to group the different repetitions and folds of the parameters predicted into a single file.

### Metrics

The metrics are computed using the models trained on the training/validation set and the test set. The metrics are saved in subfolders within the `output` folder. All the scripts have parameters that can be provided as arguments.

**Continuous Ranked Probability Score (CRPS)**

- `compute_crps.py` : Python script to compute the CRPS of the reference models and the U-Net-based methods. Ouputs are saved in the `output/{model}/CRPS` folder with `{model}` is `reference_models` or `unet_models`.
- `plot_crps.py` : Python script to plot the CRPS of the reference models and the U-Net-based methods. Outputs are saved in the `output/plots/CRPS` folder.
- `plot_crpss_raw.py` : Python script to plot the Continuous Ranked Probability Skill Score (CRPSS) of the reference models and the U-Net-based methods with respect to the raw ensemble. Outputs are saved in the `output/plots/CRPSS_raw` folder.
- `plot_crpss_qrf.py` : Python script to plot the Continuous Ranked Probability Skill Score (CRPSS) of the TQRF models and the U-Net-based methods with respect to the best QRF. Outputs are saved in the `output/plots/CRPSS_qrf` folder.

**Rank Histograms**

- `compute_rank_histograms.py` : Python script to compute the rank histograms of the reference models and the U-Net-based methods. Outputs are saved in the `output/{model}/RankHistograms` folder with `{model}` is `reference_models` or `unet_models`.
- `plot_rank_histograms.py` : Python script to plot the rank histograms of the reference models and the U-Net-based methods. Outputs are saved in the `output/plots/RankHistograms` folder.

**Receiver Operating Characteristic (ROC) curve**

- `plot_roc.py` : Python script to plot the ROC curve of the reference models and the U-Net-based methods. Outputs are saved in the `output/plots/ROC` folder.  


## References and dependencies

Here is a non-exhaustive list of the libraries and references used in this repository:
- [scoringRules](https://github.com/FK83/scoringRules) : R package to compute scoring rules.
- [ranger](https://github.com/imbs-hl/ranger) : R package providing a fast implementation of random forests.
- [cartopy](https://scitools.org.uk/cartopy/docs/latest/) : Python package for cartographic data visualization.
- [reticulate](https://rstudio.github.io/reticulate/) : R package providing interoperability between Python and R
- [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) : Python libraries for deep learning.

## Feedback

If you have any questions or feedback, please do not hesitate to inform us by opening an issue on this repository. I will do my best to answer your questions and improve the code if necessary.
