Qiskit Fall Fest: Quantum Portfolio Optimization

This repository contains a clean, commented, and organized version of the project we developed during the Qiskit Fall Fest.


The goal of our work was to tackle a financial portfolio optimization problem (selecting the 4 most profitable assets from an available pool of 8) using quantum computing. To achieve this, we implemented the QAOA (Quantum Approximate Optimization Algorithm).

Over the three days of the hackathon, we took the project through several phases:

Ideal Simulation: Definition of the QUBO problem and noiseless mathematical testing locally.

Execution on Physical Hardware: We transpiled and sent our circuit to the real 156-qubit IBM Heron processor (ibm_basquecountry), provided by Ikerbasque.

Noise Analysis: We dealt with the quantum noise of the NISQ era. We applied classical post-processing to clean up measurement errors, successfully rescuing a 23.80% feasibility rate (samples that strictly met our 4-asset constraint) and successfully finding the optimal portfolio.

 Repository Structure

data/: Contains the JSON files with the financial parameters of the assets (e.g., parametros_8_activos.json).

src/: Clean Jupyter Notebooks detailing the step-by-step execution of the experiment (from simulation to sending jobs to IBM Quantum).

results/: Raw results downloaded from the physical hardware (hardware_counts_*.json) for fidelity analysis.

 Team & Credits

This work was the result of a joint effort from Wednesday to Friday. I want to give full credit to my teammates, with whom I had an excellent working dynamic. Thank you for the laughs, the patience, and the hard work:

Ibai Gonzalez - [https://github.com/Ibaigont]

Jon Ander Duque - [https://github.com/JonAnderDS]

Iñigo Zaballa - [https://github.com/inigozaballa99-hub]

Fiona - [https://github.com/fiomcm]

Oier Zabalegi

We also want to thank the event organizers, IBM Quantum, and Ikerbasque for giving us access to this cutting-edge technology.

