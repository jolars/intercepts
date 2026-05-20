# TODO

Gaps in the current draft of `intercepts.qmd`, in rough priority order.

## Reproducibility plumbing

- [ ] Flip the fast internal-solver experiments back into the notebook so they
      run at render time, moving toward Computo's direct-reproducibility ideal.
      **Deferred to submission-prep** (the paper is still `draft: true`): the
      conversion adds real render time and there is no cheap shortcut during
      active writing (per-chunk `cache:` is a knitr/Jupyter feature the native
      `julia` engine ignores, and Quarto `freeze` is whole-document, so any edit
      re-executes everything; `_freeze` is also gitignored). The groundwork is
      done, so this is now a mechanical step:
      - Targets (each maps 1:1 to a script, a `.jld2`, and a figure; all
        synthetic, deterministic, no external data). Measured warm compute
        (post-compile):
        - `fig-per-pass-cost`/`fig-per-pass-progress` ← `sim-per-pass-cost.jl`
          (6 s)
        - `fig-cold-start-poisson` ← `sim-cold-start-poisson.jl` (17 s)
        - `fig-multinomial-imbalance` ← `sim-multinomial-imbalance.jl` (22 s)
        - `fig-irls-comparison` ← `sim-irls-comparison.jl` (26 s)
        - `fig-mu-extreme` ← `sim-mu-extreme.jl` (60 s)
        - `fig-cold-start` ← `sim-cold-start.jl` (92 s)
        - Total added render compute: \~223 s.
      - Lift recipe per figure: add `using Intercepts` and `using DrWatson` to
        the `setup` chunk (drop the now-redundant local `using Intercepts` in
        `@fig-parametric`), then replace the chunk's
        `dd = JLD2.load(...); df = DataFrame(dd["results"])` preamble with the
        script's loop verbatim (its `param_dict` + `dict_list` + per-config
        `Random.seed!` + `d_exp` pushes, minus the `using`/`@save` lines) ending
        in `df = DataFrame(results)`. Lifting verbatim (not rewriting the loop)
        reproduces the cached DataFrame row-for-row, so the existing
        `flatten(...)` + AlgebraOfGraphics plotting code stays untouched. Notes:
        `fig-cold-start-poisson` keeps its `try/catch`; `fig-irls-comparison`
        needs the `run_method` helper + `method_label`; `fig-per-pass-cost`
        lifts the full custom `cdsolver`/`inner_steps`/`relgap_after` loop and
        `fig-per-pass-progress` reuses its `df`.
      - Then delete each inlined `experiments/<script>.jl` +
        `results/<name>.jld2`, and re-verify the hard-coded prose numbers in the
        per-pass section (`6/4/3`, `1.30/1.50/1.59×`, pass counts `109/107/121`
        and `109/108/120`) against the recomputed figures.
      - Keep cached regardless: the genuinely heavy sweeps (`sim-mu-reg-heatmap`
        --- the @sec-imbalance-reg heatmaps, `sim-multinomial-imbalance-sweep`,
        `sim-warmstart-path`, `sim-rate-gap`), every real-data figure (network /
        LIBSVMdata or R dependency), and all production-solver CSVs --- the CI
        render job installs Julia only, so R/Python/network work can never run
        there.
      - Done already: removed three orphan artifacts the notebook never loaded
        (`sim-logreg-mu`, `sim-logreg-reg`, `sim-h00-heatmap`, scripts +
        `.jld2`).

## Referee-pass follow-ups

From a Computo-referee read of the current draft. Major items first, then minor
fixes. Items tagged `(verified)` were checked against the working tree; the rest
are analysis or presentation judgments to weigh.

### Claims and evidence (major)

- [x] State once, prominently (Discussion or @sec-imbalance-reg), that
      @prp-rate-gap is validated at the level of scaling and monotonicity, not
      magnitude: the empirical $T_G/T_N$ runs a constant factor of roughly 2 to
      4 below the bound throughout (@fig-rate-gap, @fig-rho-centering,
      @fig-mu-reg-gradient). The caveat is currently scattered; consolidate it
      so the divergence curves are not read as tight predictions.
- [x] Flag the single-operating-point nature of the production-solver comparison
      in the @sec-production-solvers prose. Each panel (@fig-real-glmnet through
      @fig-real-adelie) tests one $(\mu_0, \lambda)$ per problem, against the
      30-cell grid for the in-house predictions (@fig-mu-reg-gradient). Either
      say so explicitly, or sweep $\mu_0$ for the clean within-package
      glmnet/biglasso toggles.

### Data provenance (major)

- [x] Ship the real-data inputs instead of fetching them at run time.
      `experiments/fetch-yeoh.R` writes Yeoh2002 into a gitignored `data/yeoh/`,
      and Shuttle / w1a / news20 load via `LIBSVMdata.jl` and external sources.
      Computo expects all necessary data to ship (Zenodo or a pinned versioned
      archive): deposit the exact inputs and document retrieval in the README.
      The committed `.jld2` / `.csv` caches are fine for the heavy external
      phase; it is the inputs behind them that must be retrievable.

### Cross-references and presentation (minor)

- [x] Convert the four literal section references to `@sec-*` cross-references,
      adding anchors where missing: `§Results` (lines 720, 2838),
      `§Production       solvers` (956), `§Theory` (1678). The literal `§` forms
      do not render as links and can drift from the heading text. (verified)
- [x] Fix the @fig-rho-centering caption (line 922): it writes the literal
      `eq-rate-gap` instead of `@eq-rate-gap`, so it does not render as a link.
      The legend string "Predicted (eq-rate-gap)" at line 939 is a plot label;
      leave it. (verified)
- [x] Consider thinning the densest figures against Computo's "uncluttered"
      preference: @fig-multinomial-sweep (16 panels) and the crossed-factor 2x2
      grids @fig-warm-start / @fig-warm-start-mechanism, e.g. a representative
      slice in-text with the full grid in an appendix. Not blocking.

### Environment and package code (minor)

- [x] `README.qmd:56` says "Julia 1.11", but `Project.toml` (`julia = "1.12"`)
      and both CI workflows pin 1.12.6. Align the README so a re-runner picks
      the right toolchain. (verified)
- [x] Fix the dead loss default: `src/cdsolver.jl:9` and `src/gdsolver.jl:10`
      default `lossfun::LossFunction = Quadratic()`, but the type is
      `QuadraticLoss` and `Quadratic()` is undefined. It is never triggered (all
      call sites pass `lossfun` explicitly), so results are unaffected, but it
      is a dead default in a package Computo archives and assesses. Change to
      `QuadraticLoss()`. (verified, both files)
