<div align="center">

<img src="logo_artus.png" height="130px">

# Predict geospatial data with artificial intelligence

</div>

[![Documentation Status](https://readthedocs.org/projects/artus/badge/?version=latest)](https://artus.readthedocs.io/en/latest/?badge=latest)
[![Python package](https://github.com/6tronl/artus/actions/workflows/main.yml/badge.svg)](https://github.com/6tronl/artus/actions/workflows/main.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7852855.svg)](https://doi.org/10.5281/zenodo.7852855)

Artus is a python package to automatically produce maps thanks to deep learning models. With artus, you can train deep learning learning models (neural network)
on raster images annotated with vector files. You can then use the trained model to predict spatial occurrences on new unlabeled rasters. Predictions can be exported
to a GeoJson format and uploaded in your favourite GIS software.

To handle large raster file, artus provides a way to tile raster into smaller tiles according to different cutting grids.

Artus has already been implemented in three use cases using 3 differents inputs data : satellite images to detect gillnets vessels, orthomosaics to detect corals
species and under water images marked with a georeferenced point to detect marine species.


## Installation
### Prerequisites

To install jacques in your working environnement, you should already have installed a pytorch (and torchvision) version adapted to your machine resources. By following the link below, you can find the optimal configuration to install pytorch in your working environnement. Make sure to install the last version available.
> https://pytorch.org/

### Jacques installation

Jacques can be installed by executing the following lines in your terminal:

```
git clone https://gitlab.ifremer.fr/jd326f6/jacques.git
cd jacques
pip install .
```

:bulb: If you are working in an environnement, don't forget to `pip install ipykernel` to make your environnement visible in your favourite IDE.

