# Code Examples (in progress)


## **Diffpack**

This code example is the basis of many of my projects.
It is used to extract functions that describe the internal structure of protons and neutrons.
We model this internal structure, calculate its relation to experimental measurements, and perform a $\chi^2$ minimization to match our model to the experimental data.

It consists of the following modules:
* <ins>database</ins>: Experimental data is stored here
* <ins>fitlib</ins>: This module contains the central code that performs the $\chi^2$ minimization
* <ins>qcdlib</ins>: This module is where the functions are modeled
* <ins>obslib</ins>: This module relates the functions to the experimental measurements
* <ins>tools</ins>: This module contains tools used elsewhere, including parallelization and reading Excel files


## **analysis**

This directory contains the code used to submit the $\chi^2$ minimizations and analyze them afterwards.  It contains the following folders:
* <ins>inputs</ins>: Contains input files that specify the theory, model, experimental, and parameterization inputs for each $\chi^2$ minimization
* <ins>results</ins>:  Where the results are stored after the $\chi^2$ minimization
* <ins>analysis</ins>: Contains code that allows one to analyze the resulting fit to experimental data as well as the extracted functions
* <ins>plots</ins>:  Contains code used to visualize data and create figures ready for publication

Here are some example plots made in this directory:

![plot](./analysis/plots/thesis/gallery/DIS-proton.png)

<b> Caption </b>: This plot shows the comparison between my theoretical model with fitted parameters and experimental data for the experimental process known as Deep Inelastic Scattering (DIS), where a high energy electron scatters off (in this case) a proton target.  The theory result was derived in collaboration with the Jefferson Lab Angular Momentum Collaboration (JAM) and is labeled as JAM. The experimental data come from the Bologna-CERN-Dubna-Munich-Saclay (BCDMS) collaboration [[1]][BCDMS], Stanford Linear Accelerator Center (SLAC) [[2]][SLAC], New Muon Collaboration (NMC) [[3]][NMC], and Hadron-Electron Ring Accelerator [[4]][HERA].

![plot](./analysis/plots/seaquest/gallery/PDFs.png)

<b> Caption </b>: This plot shows the parameterized functions after they have been fit to the experimental data.  The functions shown here are known as unpolarized Parton Distribution Functions (PDFs), which describe the 1-dimensional momentum distribution of partons (quarks and gluons) within an unpolarized proton. The top left plot shows the up $u_v$ and down ($d_v$) quarks, the top right plot the gluon $g$, the bottom left plot the anti-up and anti-down ($\bar{d}$) quarks, and the bottom right plot the strange and anti-strange quarks.  They are a function of $x$, the parton's momentum relative to the proton's momentum. The result is compared to those of other analyses, including NNPDF3.1 [[1]][NNPDF3.1], ABMP16 [[2]][ABMP16], CJ15 [[3]][CJ15], and CT18 [[4]][CT18].

![plot](./analysis/plots/star/gallery/spin.png)

<b> Caption </b>: This plot shows the (truncated) contribution to the proton's overall spin from the Compared to other analyses: NNPDFpol1.1 [[1]][NNPDFpol1.1] and DSSV08 [[2]][DSSV08]

[BCDMS]: https://inspirehep.net/record/276661?ln=en
[SLAC]:https://inspirehep.net/literature/319089
[NMC]: http://inspirehep.net/record/424154?ln=en
[HERA]: https://inspirehep.net/record/1377206?ln=en

[NNPDF3.1]: https://inspirehep.net/literature/1602475
[ABMP16]: https://inspirehep.net/literature/1510074
[CJ15]: https://inspirehep.net/literature/1420566
[CT18]: https://inspirehep.net/literature/1773096

[NNPDFpol1.1]: https://inspirehep.net/literature/1302398
[DSSV08]: https://inspirehep.net/literature/818692







