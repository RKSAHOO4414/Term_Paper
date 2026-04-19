# Batch Edge Updates for Fully Dynamic APSP in MPC - Simulator

This repository contains the discrete-event simulator used in the experimental evaluation of the paper:

**"Batch Edge Updates for Fully Dynamic All-Pairs Shortest Paths in the MPC Model"**

## Overview

The simulator models the round complexity of the batch deletion algorithm (Algorithm 2) in the Massively Parallel Computation (MPC) model. It counts communication rounds required for processing a batch of edge deletions and compares against a sequential baseline.

## Build Instructions

### Using CMake

```bash
mkdir build
cd build
cmake ..
make
