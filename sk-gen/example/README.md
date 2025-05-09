# SimKube Example Scripts

This directory contains example scripts and resources for the SimKube project.

## Sample JQ Scripts

The `sample_scripts` directory contains example JQ scripts that can be used with the `--import-scripts` parameter:

- `double_memory.jq` - Doubles the memory allocation in container resources
- `halve_memory.jq` - Halves the memory allocation in container resources (with a minimum threshold)
- `increment_replica.jq` - Increments the replica count by 1
- `decrement_replica.jq` - Decrements the replica count by 1 (with a minimum threshold of 1)
- `double_cpu.jq` - Doubles the CPU allocation in container resources
- `halve_cpu.jq` - Halves the CPU allocation in container resources (with a minimum threshold)

## Usage

To use these scripts with sk-gen:

```bash
cargo run --bin sk-gen -- --input-traces /path/to/traces --import-scripts /path/to/sk-gen/example/sample_scripts
```

## Creating Custom Scripts

You can create your own JQ scripts for transforming Kubernetes resources. Each script should:

1. Be a valid JQ script with a `.jq` extension
2. Return an array of transformed Kubernetes resource objects
3. Be placed in a directory that is passed to the `--import-scripts` parameter

The filename (without extension) will be used as the action name in the generated traces. 