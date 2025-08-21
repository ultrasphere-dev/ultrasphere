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

# structure of spherical coordinates
c = us.c_spherical()
```

```python
# get spherical coordinates from euclidean coordinates
spherical = c.from_euclidean(torch.asarray([1.0, 2.0, 3.0]))
print(spherical)
```

```text
{'r': tensor(3.7417), 'phi': tensor(1.1071), 'theta': tensor(0.6405)}
```

```python
# get euclidean coordinates from spherical coordinates
euclidean = c.to_euclidean(spherical)
print(euclidean)
```

```text
{0: tensor(1.), 1: tensor(2.0000), 2: tensor(3.)}
```

### Using various spherical coordinates

```python
c = us.polar() # polar coordinates
c = us.c_spherical() # spherical coordinates
c = us.standard(3) # bba coordinates
c = us.standard_prime(4) # b'b'b'a coordinates
c = us.hopf(3) # ccaacaa coordinates
c = us.from_branching_types("cbab'a")
c = us.random(10)
print(f"{c.branching_types_expression_str} coordinates")
```

### Drawing Vi

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
