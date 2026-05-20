# CoSy

<div align="center">

<img src="https://raw.githubusercontent.com/tudo-seal/cosy/main/docs/assets/images/logo.svg" alt="CoSy logo" width="400" role="img">

|          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Package  | [![PyPI - Version](https://img.shields.io/pypi/v/combinatory-synthesizer.svg?logo=pypi&label=&labelColor=grey&logoColor=gold&style=flat-square)](https://pypi.org/project/combinatory-synthesizer) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/combinatory-synthesizer.svg?logo=python&label=&labelColor=grey&logoColor=gold&style=flat-square)](https://pypi.org/project/combinatory-synthesizer)                                                                                                                              |
| License  | [![License](https://img.shields.io/github/license/tudo-seal/cosy?color=9E2165&logo=apache&label=&labelColor=grey&style=flat-square)](https://opensource.org/licenses/Apache-2.0)                                                                                                                                                                                                                                                                                                                                                                     |
| Package  | [![CI - Test](https://img.shields.io/github/actions/workflow/status/tudo-seal/cosy/checks.yml?label=checks&style=flat-square)](https://github.com/tudo-seal/cosy/actions/workflows/checks.yml) [![CD - Release CoSy](https://img.shields.io/github/actions/workflow/status/tudo-seal/cosy/release.yml?label=release&style=flat-square)](https://github.com/tudo-seal/cosy/actions/workflows/release.yml)                                                                                                                                             |
| Docs     | [![Docs - Release](https://img.shields.io/github/actions/workflow/status/tudo-seal/cosy/check-docs.yml?label=checks&style=flat-square)](https://github.com/tudo-seal/cosy/actions/workflows/check-docs.yml) [![Docs - Checks](https://img.shields.io/github/actions/workflow/status/tudo-seal/cosy/deploy-docs.yml?label=deploy&style=flat-square)](https://github.com/tudo-seal/cosy/actions/workflows/deploy-docs.yml)                                                                                                                             |
| Coverage | [![codecov](https://img.shields.io/codecov/c/github/tudo-seal/cosy/main?label=main&token=40E83ABJV4&logo=codecov&labelColor=grey&style=flat-square)](https://codecov.io/github/tudo-seal/cosy/tree/main) [![codecov](https://img.shields.io/codecov/c/github/tudo-seal/cosy/develop?label=develop&token=40E83ABJV4&logo=codecov&labelColor=grey&style=flat-square)](https://codecov.io/github/tudo-seal/cosy/tree/develop)                                                                                                                           |
| Traits   | [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg?style=flat-square)](https://hatch.pypa.io/latest/) [![Checked with mypy](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftudo-seal%2Fcosy%2Fmain%2Fdocs%2Fassets%2Fbadges%2Fmypy.json&style=flat-square)](http://mypy-lang.org/) [![Checked with Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&color=4051b5&style=flat-square)](https://github.com/astral-sh/ruff) |

</div>

-----
`CoSy` enables synthesis of arbitrary artifacts from individual modular components. 
It efficiently handles specification and constraints of these modular components, 
describing how they connect and which performance criteria need to be satisfied.

## APIs

`CoSy` can be used in two different ways.

- Using the `Synthesizer`. This enables using all features but is more complicated to use. 
- Using the `Maestro`. This enables using less features, but is easy to use. 

The `Synthesizer` is the recommended way for "power-users" to interact with `CoSy`. 
Publications that primarily focus on type-theoretic aspects usually use it. 

The `Maesto` is the cute creature playing with building blocks (modular components) on the logo. 
This gifted architect is incredible at connecting these to satisfy any `target` a user may `query` for. 
The `Maestro` API is intended to be easy to use, but the trade-off is lower flexibility. 

For most technological applications of combinatory synthesis to other fields, e.g. synthesizing physical structures, 
the `Maestro` is sufficient. 

## Examples

- For a simple example for a theoretically minded computer scientist, see: [Fibonacci](https://tudo-seal.github.io/cosy/quick-start/)
- For a simple example for a practically minded engineer, see: [Robot Arm (WIP)](#)

While the examples above use `Maestro`, the following example uses the `Synthesizer`:
- For a simple example on the usage of evolutionary algorithms for searching the synthesized solution spaces, see: [Symbolic Regression](./examples/example_symbolic_regression.py)


## Installation
Installation is as simple as running: 

```console
pip install --pre combinatory-synthesizer
```

Since `CoSy` is still in pre-release state, `PyPi` distributions are likely to be outdated most of the time. 

If you want to stay up to date with a nightly build: 

```console
pip install https://github.com/tudo-seal/cosy/releases/download/nightly/combinatory_synthesizer-nightly.tar.gz
```

`CoSy` itself has no dependencies at all, so it will play nice with any pre-existing projects.

## Documentation
This README is intentionally left brief.  
Please head over to the [documentation](https://tudo-seal.github.io/cosy/) to [get started](https://tudo-seal.github.io/cosy/quick-start/). 

## License

`CoSy` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.

