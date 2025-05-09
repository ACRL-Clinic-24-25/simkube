# SK-gen

The `sk-gen` crate contains the code deliverable for the 2024-2025 Applied Computing Research Laberatories Clinic team.

## Installation

1. Install rust according to the [Installation Guide] of the Rust book.
2. Clone this repo.
3. `cargo run --release sk-gen -- --help` for usage instructions, reproduced below.

```
Usage: sk-gen [OPTIONS] --num-samples <NUM_SAMPLES> --trace-length <TRACE_LENGTH>

Options:
  -n, --num-samples <NUM_SAMPLES>
          Number of synthetic traces to generate
  -l, --trace-length <TRACE_LENGTH>
          Maximum length (in events) of each generated trace
  -e, --enumeration-steps <ENUMERATION_STEPS>
          Number of breadth-first enumeration layers to explore when expanding the state-space [default: 3]
  -i, --input-traces <INPUT_TRACES>
          Paths to messagepack files containing observed simkube traces to seed the simulation
  -v, --verbosity <VERBOSITY>
          Logging verbosity level (`trace`, `debug`, `info`, `warn`, `error`) [default: info]
      --contraction-strength <CONTRACTION_STRENGTH>
          Fraction of nodes to contract during the graph-contraction stage (range 0.0–1.0) [default: 0.5]
  -h, --help
          Print help
  -V, --version
          Print version
```


### Example invocation:
```
cargo r -r -p sk-gen -- -i trace_short.mp -e 8 --trace-length 8 --num-samples 100 -v error
```
