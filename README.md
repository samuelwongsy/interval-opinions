# Interval Opinions
This is an implementation of the paper: [These Polar Twins: Opinion Dynamics of Intervals](https://people.scs.carleton.ca/~alantsang/files/polartwins19.pdf).

## Environment
Python 3.8.9

``requirements.txt`` contains the dependencies needed for this library.

## Makefile
``make initialize``: Create and activate the python virtual environment, and create the folders for the results.

``make example``: Run ``example.py`` in the ``script`` folder. Default type of opinions is ``CoupledNetworkCastorAndPollux``, with 6 pairs of opinions, and in 3 dimensions.

``make visualize``: Run ``visualize.py`` in the ``script`` folder. Can visualize only 2 and 3 dimensions.

``make clean``: Removes the result files and visualizations.

``make all``: Runs ``initialize``, ``example``, ``visualize``.

``make run``: Runs ``example``, ``visualize``.