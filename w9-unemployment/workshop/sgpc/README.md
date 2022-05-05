SGPC
=========

This example computes a textbook stochastic growth model with irreversible investment.
This also known as a model with putty-clay capital.

That is, investment flow must be non-negative. 

This renders an ocassionally-binding constraint problem. 

Expect that this constraint tends to bind in states with low capital and low productivity.

Here we illustrate a time-iteration solution method.

This finds the fixed point (policy function) satisfying the system of nonlinear FOCs.

We solve this example using a version of a finite-element method on local hierarchical sparse grids.

Files:

* ``sgpc.py`` is the class file

* ``main.py`` is the main script for executing an example

* ``Stochastic Growth and Irreversible Investment.ipynb`` Example Jupyter Notebook

* ``Stochastic Growth and Irreversible Investment-Grid Experiments.ipynb`` Error analyses on grids and (max) local-polynomial orders

Things to do:

* Alternative version with Markov chain shocks instead of AR(1)

* Speedups comparisons using alternatives:

	* NUMBA
	
	* OpenMPI

Dependencies:

* TASMANIAN. See:

	* [website](https://tasmanian.ornl.gov/) 
	* [website for Python interface](https://pypi.org/project/Tasmanian/)
	* Install using ``pip install Tasmanian --user``

* STATSMODEL. See [website](https://www.statsmodels.org/)

(c) 2020++ T. Kam (tcy.kam@gmail.com)
