
.. _get-started:

Getting Started
===============

This guide will help you get started with installing and using IntLevPy.

Installation
------------

To get started, clone the repository and install the package using `pip`.

.. code-block:: bash

   git clone https://github.com/shailendrabhandari/IntLevy-Processes.git
   cd IntLevy-Processes
   pip install -e .

This installs the package in *editable mode*, allowing you to modify the source code without reinstalling the package.

Dependencies
------------

Make sure to install all required dependencies from the `requirements.txt` file:

.. code-block:: bash

   pip install -r requirements.txt

Usage
-----

Once installed, you can start by exploring the example scripts provided in the `examples/` directory. These scripts demonstrate core functionalities like simulating intermittent processes and Lévy flights.

### Example: Simulating an Intermittent Process

To simulate an intermittent process, run the following script:

.. code-block:: bash

   python examples/run_intermittent_simulation.py

This script generates a synthetic trajectory, calculates statistical moments, performs classification, and saves the results.

### Example: Simulating Multiple Lévy Flights

For Lévy flight simulations, run:

.. code-block:: bash

   python examples/run_levy_simulation.py

This example demonstrates how to generate and analyze multiple Lévy trajectories.

Next Steps
----------

To learn more about specific modules, refer to:
- :ref:`features` for a breakdown of the package features.
- :ref:`api/modules` for the full API documentation.

