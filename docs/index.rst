.. artus documentation master file, created by
   sphinx-quickstart on Thu Jun  1 14:39:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: ../figures/logo_artus.png
   :height: 100

Welcome to artus's documentation!
=================================

|Read the Docs| |Python package| |DOI|

Artus is a python package to automatically produce maps thanks to deep learning models. With artus, you can train deep learning learning models (neural network)
on raster images annotated with vector files. You can then use the trained model to predict spatial occurrences on new unlabeled rasters. Predictions can be exported
to a GeoJson format and uploaded in your favourite GIS software.

To handle large raster file, artus provides a way to tile raster into smaller tiles according to different cutting grids.

Artus has already been implemented in three use cases using 3 differents inputs data : satellite images to detect gillnets vessels, orthomosaics to detect corals
species and under water images marked with a georeferenced point to detect marine species.

.. image:: ../figures/gillnets_X101_predictions.png
   :alt: Detection of gilnets vessels on satellite images
   :height: 150

.. image:: ../figures/predicted_species_ngouja.png
   :alt: Detection of coral species on under water orthomosaics
   :height: 150

.. image:: ../figures/heatmap_study_area.png
   :alt: Detection of dead corals on georeferenced images
   :height: 150


Installation
-----------

.. toctree::
   :caption: Installation

   installation


This project is being developed as part of the G2OI project, cofinanced by the European union, the Reunion region, and the French Republic.

.. image:: ../figures/logo_partenaires.png
   :alt: Financials partners of this project : European union, the Reunion region, and the French Republic
   :height: 40


How to cite
-----------

.. epigraph::

   Justine Talpaert Daudon. (2023). artus v0.X: A tool to automatically procude mamps with artificial intelligence.(v0.X). Zenodo. DOI:10.5281/zenodo.7852855

.. toctree::
   :maxdepth: 3
   :caption: Modules

   modules
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
