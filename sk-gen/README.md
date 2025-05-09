# SK-gen

The `sk-gen` crate contains the code deliverable for the 2024-2025 Applied Computing Research Laboratories Clinic team.

## Getting started
1. Install rust according to the [Installation Guide] of the Rust book.
2. Clone this repo.
3. Read CLI-help (instructions reproduced below)
4. Generate and read the crate documentation (instructions reproduced below)
5. Try running against the provided example traces and scripts (instructions reproduced below)

## Usage Examples
### Help invocation:
```
cargo run --package sk-gen -- --help
```
```
Usage: sk-gen [OPTIONS] --num-samples <NUM_SAMPLES> --trace-length <TRACE_LENGTH>

Options:
  -n, --num-samples <NUM_SAMPLES>
          Number of synthetic traces to generate
  -l, --trace-length <TRACE_LENGTH>
          Maximum length (in events) of each generated trace
  -e, --enumeration-steps <ENUMERATION_STEPS>
          Number of breadth-first enumeration layers to explore when expanding the state-spa
ce [default: 3]
  -t, --input-traces <INPUT_TRACES>
          Paths to files containing `SimKube` `ExportedTraces` serialized as either JSON or
`MessagePack`
  -v, --verbosity <VERBOSITY>
          Logging verbosity level (`trace`, `debug`, `info`, `warn`, `error`) [default: info
]
      --contraction-strength <CONTRACTION_STRENGTH>
          Fraction of nodes to contract during the graph-contraction stage (range 0.0–1.0) [
default: 0.5]
  -s, --script-directory <SCRIPT_DIRECTORY>
          Directory containing JQ scripts to import (format: {name}.jq)
  -h, --help
          Print help
  -V, --version
          Print version
```

### Generate and open documentation
```
cargo doc --package sk-gen --document-private-items --no-deps --open
```

### Example run:
```
cargo run --release --package sk-gen -- --input-traces sk-gen/example/sample_traces/dsb-2- services.json --enumeration-steps 4 --trace-length 8 --num-samples 100 --verbosity warn --script-directory sk-gen/example/sample_scripts/
```
```
  [00:00:00] [████████████████████████████████████████] 9/9 nodes (100%) Layer 1/4
  [00:00:00] [████████████████████████████████████████] 77/77 nodes (100%) Layer 2/4
  [00:00:00] [████████████████████████████████████████] 403/403 nodes (100%) Layer 3/4
  [00:00:01] [████████████████████████████████████████] 1433/1433 nodes (100%) Layer 4/4
  [00:00:53] [████████████████████████████████████████] 3033/3033 contractions (100%) Contra ction phase (50% of nodes)
  [00:00:02] [████████████████████████████████████████] 100/100 traces (100%) Generating tra ces (length 8)
```

Following the execution, inside `runs/{latest_timestamp}` you will find:
- Serialized graphs at each stage for visualization in other software at *.dot
- traces in at trace_*.json, these are serialized values of the ExportedTrace type.
