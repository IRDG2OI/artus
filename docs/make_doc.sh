pip uninstall artus
pip install ../
rm ./modules.rst
rm ./artus.rst
rm ./artus.evaluate_model.rst
rm ./artus.inference.rst
rm ./artus.prepare.rst
rm ./artus.spatialize.rst
rm ./artus.train.rst

sphinx-apidoc -o . ../src
make html