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

## Referee-pass follow-ups (2026-05-20)

From a fresh Computo-referee read of the current draft. The previous referee-pass
items are all addressed and have been cleared. Major items first, then minor
fixes. Items tagged `(verified)` were checked against the working tree; the rest
are analysis or presentation judgments to weigh.

### Reproducibility (major --- the necessary condition)

- [ ] Publish the Zenodo deposit. `experiments/fetch-data.sh:21` and
      `README.qmd:74` pin DOI `10.5281/zenodo.20315625`, but it is a *reserved*
      DOI --- the record 404s, so a referee following the README cannot fetch
      `intercepts-data-v1.tar.gz` (gitignored, not committed). The committed
      caches under `results/` let the notebook render, but Computo's
      reproducibility condition covers re-running, and "all necessary data (e.g.
      via Zenodo)" must resolve. Build the archive (`build-data-archive.jl`),
      publish the deposit, and confirm `data/MANIFEST.sha256` verifies against
      it. Until then, document the `INTERCEPTS_DATA_ARCHIVE=...` local override
      as the working path. (verified: DOI reserved, archive gitignored)
- [x] Reconcile the README with the actual environment files. `README.qmd:117`
      and `:121` tell the re-runner to `nix develop` against a `flake.nix`, but
      the repo ships `devenv.nix` / `devenv.lock` and there is no root
      `flake.nix` --- the documented path dead-ends. Point the README at the
      real files (or add the flake). (verified: no `flake.nix`) (resolved:
      rewrote the README section to point at `devenv.nix`/`devenv.lock`, enter
      with `devenv shell`, and link to the devenv install docs; fixed the same
      stale `nix develop` in `AGENTS.md`. Regenerate `README.md` with
      `task readme`.)

### Claims and framing (major)

- [ ] Reconsider the "Proposition" label on @prp-rate-gap. As written
      (@eq-rate-gap, @eq-rate-gap-asymptotic) it is a ratio of upper bounds built
      from three assumption-violating substitutions, and the empirical ratio
      runs a constant factor of 2--4 below it. The surviving content is scaling
      and monotonicity (already consolidated as a caveat), which the data
      support --- but a numbered Proposition implies a tightness it lacks.
      Downgrade to a scaling argument / remark, or justify the numbered framing.
      Distinct from the (done) caveat-consolidation item: this is about the
      structural label, not the prose hedge.
- [x] Soften the abstract's "confirm the predicted ordering across six
      production solvers." The single-operating-point caveat is already in the
      @sec-production-solvers prose, but the abstract still reads stronger than
      single-$(\mu_0,\lambda)$ panels can show. State that the scaling evidence
      is the within-package toggles, the skglm before/after patch, and the
      in-house heatmaps; the six-solver panels establish direction. (abstract
      now reads "the predicted ordering holds at the operating points tested,
      establishing the direction ... rather than its scaling"; the scaling
      claim is attributed to the within-package toggles, the before/after skglm
      patch, and in-house sweeps---mirroring the @sec-production-solvers
      direction-vs-scaling split.)

### Clarity and presentation (minor)

- [x] Make the abstract and the opening of @sec-introduction readable by a
      computational scientist who does not yet know $H_{00}$, the Schur
      complement, or IRLS. Defer the notation; lead with the phenomenon
      (gradient updates stall when the response is imbalanced) and the
      recommendation (one Newton step). Computo wants the abstract/intro as
      nontechnical as possible. (abstract in `_quarto.yml` and intro para 1--2
      rewritten: $L_0$/$H_{00}$/Armijo gone from the abstract, IRLS/prox-Newton
      deferred to intro para 2 with IRLS defined at first use; the
      Schur-complement naming stays in the contributions list, which now comes
      after the phenomenon and the strategies.)
      signpost. @tbl-real-logreg-problems is internally consistent (header and
      caption both use $H_{00}/L_0 = 4\mu_0(1-\mu_0)$; the slowdown ordering at
      line 1712 uses the reciprocal $L_0/H_{00}$), so this is signposting for a
      reader checking the arithmetic, not a bug. (verified: not a
      header/caption mismatch)
- [ ] Consider folding the two Poisson no-op demonstrations into one subsection:
      @fig-cold-start-poisson and @fig-real-poisson make near-identical points.
      The densest grids (@fig-multinomial-sweep-full, @fig-warm-start-full,
      @fig-warm-start-mechanism-full) are already in the supplement. Not
      blocking.

Not actionable from this pass: the review's "README structure boilerplate"
point does not apply --- `README.qmd:31` already uses `<experiments/>` and the
capitalized `Intercepts` module, and `README.qmd:56` already pins Julia 1.12.6.
