# Architecture Notes

This project currently solves a hybrid problem:

- The FP/KFE equation is time-dependent and is marched forward.
- The HJB is stationary in form and is re-solved along the FP path using the current distribution.

The folder layout separates this current implementation from the next step, where both
the FP/KFE and HJB should become time-dependent.

## Current Layout

```text
src/core/
    Shared model primitives: parameters, grids, derivatives, aggregates.

src/solvers/hjb_stationary.jl
    Stationary-form HJB solver used by the current hybrid algorithm.

src/solvers/fp_kfe.jl
    Forward equation policies, generator assembly, and implicit Euler update.

src/solvers/coupled_quasistatic.jl
    Current outer loop: FP forward in time, stationary HJB re-solved at each FP step.

src/visualization/
    Plotting and generated-figure helpers.
```

## Planned Direction

The future full MFG solver should add a dedicated time-dependent HJB solver rather than
mutating the stationary solver in place. A natural next split is:

```text
src/solvers/hjb_time_dependent.jl
src/solvers/coupled_forward_backward.jl
```

That lets us keep the current hybrid solver available as a benchmark while introducing
the true forward-backward algorithm.
