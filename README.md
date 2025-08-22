# ultrasphere

<p align="center">
  <a href="https://github.com/34j/ultrasphere/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/ultrasphere/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://ultrasphere.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/ultrasphere.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/ultrasphere">
    <img src="https://img.shields.io/codecov/c/github/34j/ultrasphere.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/ultrasphere/">
    <img src="https://img.shields.io/pypi/v/ultrasphere.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/ultrasphere.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/ultrasphere.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://ultrasphere.readthedocs.io" target="_blank">https://ultrasphere.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/ultrasphere" target="_blank">https://github.com/34j/ultrasphere </a>

---

Hyperspherical coordinates in NumPy / PyTorch / JAX.

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install ultrasphere
```

## Usage

### Spherical Coordinates ↔ Cartesian Coordinates

```python
import ultrasphere as us
import torch

# 1. specify the structure of spherical coordinates
c = us.c_spherical()

# 2. get spherical coordinates from euclidean coordinates
spherical = c.from_euclidean(torch.asarray([1.0, 2.0, 3.0]))
print(spherical)
# {'r': tensor(3.7417), 'phi': tensor(1.1071), 'theta': tensor(0.6405)}

# 3. get euclidean coordinates from spherical coordinates
euclidean = c.to_euclidean(spherical)
print(euclidean)
# {0: tensor(1.), 1: tensor(2.0000), 2: tensor(3.)}
```

### Using various spherical coordinates

```python
c = us.polar()  # polar coordinates
c = us.c_spherical()  # spherical coordinates
c = us.standard(3)  # bba coordinates
c = us.standard_prime(4)  # b'b'b'a coordinates
c = us.hopf(3)  # ccaacaa coordinates
c = us.from_branching_types("cbab'a")
c = us.random(10)

# get the branching types expression
print(c.branching_types_expression_str)
# ccabbab'b'ba
```

### Drawing spherical coordinates using rooted trees (Vilenkin's method of trees)

#### Python

```python
import ultrasphere as us

# 1. specify the structure of spherical coordinates
c = us.random(10)

# 2. draw the rooted tree
us.draw(c)
```

#### CLI

```shell
ultrasphere "ccabbab'b'ba"
```

Output:

![ccabbab'b'ba](https://raw.githubusercontent.com/34j/ultrasphere/main/coordinates.jpg)

### Integration over sphere using spherical coordinates

```python
import ultrasphere as us
import numpy as np

# 1. specify the structure of spherical coordinates
c = us.c_spherical()

# 2. integrate a function over the sphere
integral = us.integrate(
    c, lambda spherical: spherical["theta"] ** 2 * spherical["phi"], False, 10, xp=np
)
print(integral)
# 110.02620812972036
```

### Random sampling

```python
import ultrasphere as us
import numpy as np

# 1. specify the structure of spherical coordinates
c = us.c_spherical()

# 2. sample random points uniformly from the ball
points_ball = us.random_ball(c, shape=(), xp=np)
print(points_ball, np.linalg.vector_norm(points_ball))
# [ 0.83999061  0.02552206 -0.29185517] 0.8896151114371893

# 3. sample random points uniformly from the sphere
points_sphere = us.random_ball(c, shape=(), xp=np, surface=True)
print(points_sphere, np.linalg.vector_norm(points_sphere))
# [-0.68194186  0.71310149 -0.16260864] 1.0
```

#### References

- Barthe, F., Guedon, O., Mendelson, S., & Naor, A. (2005). A probabilistic approach to the geometry of the \ell_p^n-ball. arXiv, math/0503650. Retrieved from https://arxiv.org/abs/math/0503650v1

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
