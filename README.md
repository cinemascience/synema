# Synema

**Synema** is a collection of neural radiance field (NeRF-style)
view-synthesis utilities designed to work with **Cinema image-based
databases** for in situ / in transit visualization workflows.

The goal is to take an existing Cinema dataset (images + camera
parameters) and enable **novel view synthesis** (new viewpoints, smooth
camera paths, view interpolation) to enhance interactive exploration
without requiring additional simulation output.

------------------------------------------------------------------------

## Backend: JAX / Flax Ecosystem

Synema is implemented using the **JAX ecosystem**, not PyTorch or
TensorFlow.

Core ML stack:

-   **JAX** --- numerical backend (XLA-compiled array computing)
-   **Flax** --- neural network module system
-   **Optax** --- optimizers
-   **Jaxtyping** --- type annotations for JAX arrays

### Why JAX?

-   High-performance XLA compilation
-   Functional programming model well-suited for NeRF training
-   Clean separation between model definition (Flax) and optimization
    (Optax)
-   Excellent support for GPU and TPU acceleration

### GPU Support

To use a GPU, install the correct JAX build for your CUDA version:

``` bash
pip install --upgrade "jax[cuda12]"
```

Refer to the official JAX installation guide for CUDA-specific wheels:
https://jax.readthedocs.io/en/latest/installation.html

If no GPU is available, Synema will run on CPU (training will be
slower).

------------------------------------------------------------------------

## What's in this repository

-   **`synema/`** --- Python package implementing models, data handling,
    and rendering utilities\
-   **`examples/`** --- Example scripts demonstrating training and
    inference on Cinema datasets\
-   **`data/`** --- Example assets and/or helper data\
-   **`pyproject.toml`** --- Modern Python packaging (PEP 621
    compliant)\
-   **`requirements-dev.txt`** --- Development tooling dependencies\
-   **`requirements-examples.txt`** --- Optional example / visualization
    dependencies\
-   **`license.md`** --- Repository license

------------------------------------------------------------------------

## Requirements

Typical environment:

-   Python 3.9+
-   CUDA-capable GPU (recommended for training)
-   JAX ecosystem (jax, flax, optax)
-   Standard scientific Python libraries

------------------------------------------------------------------------

## Installation

### Core install

``` bash
pip install -e .
```

------------------------------------------------------------------------

## Optional Extras (Recommended)

Synema provides optional dependency groups defined in `pyproject.toml`.

### Development tools

Includes testing, linting, and packaging utilities:

``` bash
pip install -e .[dev]
```

Equivalent:

``` bash
pip install -r requirements-dev.txt
```

------------------------------------------------------------------------

### Examples

To run PyVista-based examples:

``` bash
pip install -e .[gui]
```

Equivalent:

``` bash
pip install -r requirements-examples.txt
```

------------------------------------------------------------------------

## Full Development Setup

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev,gui]
```

------------------------------------------------------------------------

## Quick Start

Typical workflow:

1.  Train a model on a Cinema dataset
2.  Render novel views
3.  Export results for Cinema visualization

Example (see `examples/` for actual scripts):

``` bash
python examples/train.py --cinema /path/to/cinema_db --out runs/my_run
python examples/render.py --run runs/my_run --trajectory spiral --out renders/my_run_spiral
```

------------------------------------------------------------------------

## Cinema Data Expectations

Synema expects:

-   Images from a Cinema database
-   Camera pose and intrinsics metadata
-   Parameter metadata for consistent rendering

Adapters in `synema/` or `examples/` map Cinema metadata to NeRF
training inputs.

------------------------------------------------------------------------

## Project Structure

    synema/
      synema/        # Models, datasets, rendering, utilities
      examples/      # Training and rendering examples
      data/          # Example data
      pyproject.toml # Packaging configuration

------------------------------------------------------------------------

## Contributing

Contributions welcome via issues and pull requests:

-   New model variants
-   Dataset adapters
-   Performance improvements
-   Documentation updates

------------------------------------------------------------------------

## Citation

``` bibtex
@software{synema,
  title  = {Synema: Novel View Synthesis for Cinema in situ Visualization},
  author = {CinemaScience contributors},
  url    = {https://github.com/cinemascience/synema},
  year   = {2025}
}
```

------------------------------------------------------------------------

------------------------------------------------------------------------

## AI Assistance Disclosure

Portions of this repository's documentation, packaging configuration,
and test scaffolding were generated or refined with the assistance of AI
tools (including large language models).

All generated content has been reviewed and curated by the project
maintainers. Any errors, omissions, or inconsistencies remain the
responsibility of the human authors.

------------------------------------------------------------------------

## License

See `license.md`.
