<div align="center">

<img src="https://github.com/6tronl/artus/blob/main/docs/logo_artus.png?raw=True" height="130px">

# ACKNOWLEDGEMENT

This project is being developed as part of the G2OI project, cofinanced by the European Union, the Reunion region, and the French Republic.
<div align="center">


<img src="https://github.com/IRDG2OI/geoflow-g2oi/blob/main/img/logos_partenaires.png?raw=True" height="80px">

</div>

# Predict geospatial data with artificial intelligence

[![Documentation Status](https://readthedocs.org/projects/artus/badge/?version=latest)](https://artus.readthedocs.io/en/latest/?badge=latest)
[![Python package](https://github.com/6tronl/artus/actions/workflows/main.yml/badge.svg)](https://github.com/6tronl/artus/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/artus.svg)](https://badge.fury.io/py/artus)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7852855.svg)](https://doi.org/10.5281/zenodo.7852855)

</div>


Artus is a python package to automatically produce maps thanks to deep learning models. With artus, you can train deep learning learning models (neural network)
on raster images annotated with vector files. You can then use the trained model to predict spatial occurrences on new unlabeled rasters. Predictions can be exported to a GeoJson format and uploaded in your favourite GIS software.

To handle large raster file, artus provides a way to tile raster into smaller tiles according to different cutting grids.

Artus has already been implemented in three use cases using 3 differents inputs data : satellite images to detect gillnets vessels, orthomosaics to detect corals
species and under water images marked with a georeferenced point to detect marine species.

For example, the following map is generated by automatically detecting dead corals on images associated with a single GPS point:
<div align="center">
<img src="https://github.com/6tronl/artus/blob/main/docs/heatmap_study_area.png?raw=True" height="500px">
</div>

This project is being developed as part of the G2OI project, cofinanced by the European union, the Reunion region, and the French Republic.
<img src="https://github.com/6tronl/artus/blob/main/docs/logos_partenaires.png?raw=True" height="40px">

## Installation

All the installation procedures are available here : [install artus](https://artus.readthedocs.io/en/latest/installation.html)

## Tutorials

If you want to get started with artus you can follow the [notebooks](https://github.com/6tronl/artus-examples). Depending on you requirements, you will find tutorials to train
a new deep learning model, to predict an unlabeled raster or to convert different annotations files (COCO, geojson...).

## Documentation

Documentation is available on [Read the docs](https://artus.readthedocs.io/en/latest/)



