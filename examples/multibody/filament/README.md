# Filament examples

This directory contains examples for simulating slender deformable objects (filaments) in Drake.

- For C++ examples (`.cc` files):
  1. Run the visualizer
     ```
     bazel run //tools:meldis &
     ```
  2. Run the example, e.g.,
     ```
     bazel run //examples/multibody/filament:slack_rope
     ```
- For Python examples (`.ipynb` files):
  1.  Build this repository by following the instructions [here](https://drake.mit.edu/from_source.html#building-with-cmake).
  2.  Configure the `PYTHONPATH` environment variable by following the instructions [here](https://drake.mit.edu/from_source.html#running-the-python-bindings-after-a-cmake-install).
  3. You can now try out the Jupyter notebook files.