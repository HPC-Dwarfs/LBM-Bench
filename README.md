# LBM-Bench

A Lattice-Boltzmann CFD solver and benchmark for evaluating memory-bandwidth-bound
performance on modern HPC architectures. The code implements a D3Q19 BGK collision
operator in several propagation/data-layout variants so that the achievable
memory bandwidth can be measured with different algorithmic characteristics.

## Features

- D3Q19 lattice with BGK collision
- Seven kernel variants (propagation model × data layout):
  - `push-soa`, `push-aos` — push-based propagation
  - `pull-soa`, `pull-aos` — pull-based propagation
  - `blk-push-soa`, `blk-pull-soa` — cache-blocked push/pull (SoA)
  - `aa-soa` — AA-pattern (fused propagation + collision, SoA)
- Five geometry types: `box`, `channel`, `pipe`, `blocks-N`, `fluid`
- Optional periodic boundary conditions per axis
- Single (`sp`) and double (`dp`) precision
- OpenMP threading with first-touch initialisation
- Optional LIKWID hardware-counter instrumentation
- Built-in verification against the analytical Poiseuille channel-flow profile

## Requirements

- C99-capable compiler: GCC, Clang, or Intel oneAPI `icx`
- GNU Make
- (optional) OpenMP runtime
- (optional) [LIKWID](https://github.com/RRZE-HPC/likwid) performance tools

## Building

### 1. Create `config.mk`

Running `make` for the first time automatically copies
`mk/config-default.mk` to `config.mk` and stops. Edit the file to match
your environment before building again:

```makefile
# Supported: GCC, CLANG, ICX
TOOLCHAIN ?= CLANG
# Supported: dp (double precision), sp (single precision)
PRECISION ?= dp
ENABLE_OPENMP ?= false
ENABLE_LIKWID ?= false
```

Uncomment additional `OPTIONS` lines to enable optional features:

| Option                 | Effect                                            |
| ---------------------- | ------------------------------------------------- |
| `-DVERIFICATION`       | Enable the built-in Poiseuille verification check |
| `-DVERBOSE_AFFINITY`   | Print thread-to-core binding                      |
| `-DVERBOSE_DATASIZE`   | Print allocation sizes                            |
| `-DVERBOSE_TIMER`      | Print per-iteration timing                        |
| `-DARRAY_ALIGNMENT=64` | Align allocations to 64 bytes (default)           |

### 2. Build

```sh
make                          # default toolchain / precision from config.mk
make TOOLCHAIN=GCC            # override toolchain
make PRECISION=sp             # single-precision build
make TOOLCHAIN=ICX PRECISION=dp
```

The binary is placed in the repository root and named
`lbmbench-<TOOLCHAIN>-<PRECISION>` (e.g. `lbmbench-CLANG-dp`).

### Additional make targets

| Target           | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `make clean`     | Remove build artefacts for the current toolchain/precision |
| `make distclean` | Remove all build directories and binaries                  |
| `make asm`       | Generate annotated assembly in `build/`                    |
| `make format`    | Format all sources with `clang-format`                     |
| `make test`      | Run the verification test suite (see below)                |
| `make info`      | Print compiler flags and version                           |

## Usage

```
lbmbench [options]
  -d XxYxZ             Geometry dimensions (default: 20x20x20)
  -g TYPE              Geometry type: box|channel|pipe|blocks-N|fluid
  -i N                 Number of iterations (default: 10)
  -o V                 Relaxation parameter omega (default: 1.0)
  -f V                 X-direction body force (default: 0.00001)
  -k NAME              Kernel to use (default: push-soa)
  -l                   List available kernels and exit
  -V                   Run built-in Poiseuille verification
  -x/-y/-z             Enable periodic boundary conditions in x/y/z
  -h                   Print this help
```

### Example runs

```sh
# Benchmark with default settings
./lbmbench-CLANG-dp

# 200³ domain, pull-soa kernel, 100 iterations
./lbmbench-CLANG-dp -d 200x200x200 -k pull-soa -i 100

# OpenMP (set threads via OMP_NUM_THREADS)
OMP_NUM_THREADS=8 ./lbmbench-GCC-dp -d 300x300x300 -k aa-soa -i 200

# Single precision, blocked kernel
./lbmbench-CLANG-sp -d 256x256x256 -k blk-push-soa -i 50

# List all available kernels
./lbmbench-CLANG-dp -l
```

### Output

The benchmark prints a human-readable summary and a single parseable line:

```
P:   <MFLUP/s>  d: <s>  iter: <N>  fnodes: <M x1e6>  geo: <type>  kernel: <name>
```

where **MFLUP/s** is Mega Fluid Lattice Updates per second — the standard LBM
throughput metric. Memory bandwidth is derived from the theoretical loop balance
of each kernel variant.

## Verification

Build with `-DVERIFICATION` enabled (uncomment the line in `config.mk`) and run:

```sh
./lbmbench-CLANG-dp -V -k push-soa
```

The solver runs a 16³ periodic Poiseuille flow for 1 000 iterations and compares
the resulting velocity profile to the analytical solution. The test passes when
the L2 error norm is below 0.1.

The `make test` target automates this for both precisions and all available
kernels:

```sh
make test                  # uses toolchain from config.mk
make test TOOLCHAIN=GCC
```

## LIKWID instrumentation

Set `ENABLE_LIKWID=true` in `config.mk` and add `-DLIKWID_PERFMON` to `OPTIONS`.
The main propagation kernel is wrapped in a marker region named `PROPKERNEL`.
Run with the LIKWID wrapper:

```sh
likwid-perfctr -C 0-7 -g MEM -m ./lbmbench-CLANG-dp -d 200x200x200 -i 100
```

## License

MIT — see [LICENSE](LICENSE).  
Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
