# Code Examples (in progress)


**Diffpack**

This code example is the basis of many of my projects.
It is used to extract functions that describe the internal structure of protons and neutrons.
We model this internal structure, calculate its relation to experimental measurements, and perform a $\chi^2$ minimization to match our model to the experimental data.

It consists of the following modules:
* <u>database</u>: Experimental data is stored here
* <u>fitlib</u>: This module contains the central code that performs the $\chi^2$ minimization
* <u>qcdlib</u>: This module is where the functions are modeled
* <u>obslib</u>: This module relates the functions to the experimental measurements
* <u>tools</u>: This module contains tools used elsewhere, including parallelization and reading Excel files


**analysis**


![plot](./analysis/plots/thesis/gallery/lepton-asym.png)

![plot](./analysis/plots/seaquest/gallery/PDFs.png)

![plot](./analysis/plots/star/gallery/spin.png)






