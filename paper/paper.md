---
title: "ultrasphere and ultrasphere-harmonics: Python packages for Vilenkin–Kuznetsov–Smorodinsky polyspherical coordinates and hyperspherical harmonics techniques in array API"
tags:
  - Python
authors:
  - given-names: Hiromochi
    surname: Itoh
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Author 2
    affiliation: 2
  - name: Author 3
    corresponding: true
    affiliation: 3
affiliations:
  - name: Department of Mechanical Engineering, Graduate School of Engineering, The University of Tokyo, Japan
    index: 1
    ror: 057zh3y96
  - name: Graduate School of Advanced Science and Engineering, Hiroshima University, Japan
    index: 2
    ror: 03t78wx29
  - name: Department of Strategic Studies, Institute of Engineering Innovation, Graduate School of Engineering, The University of Tokyo
    index: 3
    ror: 057zh3y96
date: 23 September 2025
bibliography: paper.bib
---

# Summary

Spherical harmonics, which are the solutions to the angular part of the laplace equation, have been widely used in various fields of science and engineering.

2-dimensional spherical harmonics are known as Fourier series .

# Statement of need

`ultrasphere` and `ultrasphere-harmonics` are Python packages for hyperspherical coordinates and hyperspherical harmonics techniques.
Our packages is that they support any type of Vilenkin–Kuznetsov–Smorodinsky polyspherical coordinate systems. This allows to write codes that work in any type of polyspherical coordinates and any number of dimensions without modification. To demonstrate this, we implemented acoustic scattering from a single sphere for any type of polyspherical coordinates, which could be verified by command-line interface.

Our api is compatible with the array API standard. This enables writing code which runs on multiple array libraries (e.g., NumPy, CuPy, JAX, PyTorch, TensorFlow) and multiple hardware (e.g., CPU, GPU) without modification. Our packages fully support vectorization for high performance computing.

# Acknowledgements

This work used computational resources
Supermicro ARS-111GL-DNHR-LCC and FUJITSU Server PRIMERGY CX2550 M7 (Miyabi) at Joint Center for Advanced High Performance Computing (JCAHPC) and
TSUBAME4.0 supercomputer provided by Institute of Science Tokyo
through Joint Usage/Research Center for Interdisciplinary Large-scale Information Infrastructures and High Performance Computing Infrastructure in Japan (Project ID: jh240031).

# References
