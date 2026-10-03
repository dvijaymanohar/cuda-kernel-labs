# Learning path

The repository specification is implemented as a sequence of controlled experiments.

## Progression

- [ ] 01 — indexing
- [ ] 02 — vector add
- [ ] 03 — grid-stride loop
- [ ] 04 — reduction
- [ ] 05 — scan
- [ ] 06 — histogram
- [ ] 07 — transpose
- [ ] 08 — shared-memory tiling
- [ ] 09 — divergence
- [ ] 10 — atomics
- [ ] 11 — streams/events
- [ ] 12 — overlap

## Evidence template

For every performance experiment, record:

- objective
- hypothesis
- workload and input shape
- independent variable
- controlled variables
- hardware/software environment
- correctness criterion
- timing methodology
- median and spread
- profiler evidence
- interpretation
- trade-off
- next experiment

Do not optimize a workload until the baseline is correct and the bottleneck hypothesis is supported by evidence.
