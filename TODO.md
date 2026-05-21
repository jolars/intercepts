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

From a fresh Computo-referee read of the current draft. The previous
referee-pass items are all addressed and have been cleared. Major items first,
then minor fixes. Items tagged `(verified)` were checked against the working
tree; the rest are analysis or presentation judgments to weigh.

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

- [x] Reconsider the "Proposition" label on @prp-rate-gap. As written
      (@eq-rate-gap, @eq-rate-gap-asymptotic) it is a ratio of upper bounds
      built from three assumption-violating substitutions, and the empirical
      ratio runs a constant factor of 2--4 below it. The surviving content is
      scaling and monotonicity (already consolidated as a caveat), which the
      data support --- but a numbered Proposition implies a tightness it lacks.
      Downgrade to a scaling argument / remark, or justify the numbered framing.
      Distinct from the (done) caveat-consolidation item: this is about the
      structural label, not the prose hedge. (resolved: relabeled the block from
      `.proposition #prp-rate-gap` to `.remark` --- Quarto renders it unnumbered
      and non-cross-referenceable, so it no longer sits beside the proven
      @prp-profile-equiv as a peer. Content unchanged. Re-pointed all 10
      `@prp-rate-gap` citations to `@eq-rate-gap` (or `@eq-rate-gap-asymptotic`
      where the text means the asymptotic limit), rewording the two sites where
      an equation could not be the grammatical agent to "the rate-gap argument";
      also fixed two prose uses of the word "proposition" that referred to this
      block. No broken refs: both equation labels still resolve.)
- [x] Soften the abstract's "confirm the predicted ordering across six
      production solvers." The single-operating-point caveat is already in the
      @sec-production-solvers prose, but the abstract still reads stronger than
      single-$(\mu_0,\lambda)$ panels can show. State that the scaling evidence
      is the within-package toggles, the skglm before/after patch, and the
      in-house heatmaps; the six-solver panels establish direction. (abstract
      now reads "the predicted ordering holds at the operating points tested,
      establishing the direction ... rather than its scaling"; the scaling claim
      is attributed to the within-package toggles, the before/after skglm patch,
      and in-house sweeps---mirroring the @sec-production-solvers
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
- [x] Reduce the $H_{00}/L_0 \leftrightarrow L_0/H_{00}$ alternation, or add a
      signpost. @tbl-real-logreg-problems is internally consistent (header and
      caption both use $H_{00}/L_0 = 4\mu_0(1-\mu_0)$; the slowdown ordering at
      line 1712 uses the reciprocal $L_0/H_{00}$), so this is signposting for a
      reader checking the arithmetic, not a bug. (verified: not a header/caption
      mismatch) (resolved: kept the slowdown framing---it is the ordering
      rationale and is used in both the prose and the @fig-real-logreg
      caption---and added a one-clause signpost at the table handoff noting the
      column lists the reciprocal $H_{00}/L_0$.)
- [x] Consider folding the two Poisson no-op demonstrations into one subsection:
      @fig-cold-start-poisson and @fig-real-poisson make near-identical points.
      The densest grids (@fig-multinomial-sweep-full, @fig-warm-start-full,
      @fig-warm-start-mechanism-full) are already in the supplement. Not
      blocking. (declined: the two figures do not make near-identical points.
      @fig-cold-start-poisson is a synthetic constructed-failure example in the
      Theory section whose protagonist is the unguarded-Newton overshoot and
      whether the Armijo guard absorbs it; @fig-real-poisson is real-data
      (congress109) confirmation in Results whose protagonist is the
      gradient-strategy no-op ($L_0=\infty$). They serve the theory→experiment
      arc, sit in different sections, and already cross-reference rather than
      repeat---@fig-real-poisson cites @fig-cold-start-poisson to explain the
      guarded/unguarded curve coincidence. Folding would relocate a synthetic
      example into the empirical section. Kept all three Poisson figures
      (@fig-real-poisson-production is the production-solver landscape, a
      distinct point).)

Not actionable from this pass: the review's "README structure boilerplate" point
does not apply --- `README.qmd:31` already uses `<experiments/>` and the
capitalized `Intercepts` module, and `README.qmd:56` already pins Julia 1.12.6.

## Referee-pass follow-ups (2026-05-21)

From another fresh Computo-referee read. Items tagged `(verified)` were checked
against the working tree; the rest are framing or presentation judgments to
weigh. Two of this pass's points are already tracked above and not repeated: the
Zenodo deposit (open major item under the 2026-05-20 reproducibility section),
and the abstract/intro nontechnical rewrite (done) --- the referee re-flagged
the notation in the contributions list, but keeping the Schur-complement naming
there was a conscious call recorded above.

### Reproducibility (major --- the necessary condition)

- [ ] Document the CI split in @sec-methodology. `build.yml` only renders the
      paper (it installs Julia, runs `quarto render`, deploys Pages) and never
      runs the test suite; `test.yml` runs `Pkg.test()` separately on push / PR.
      Since Computo treats CI as a reproducibility instrument and `src/` quality
      is under review, state plainly which workflow guarantees what, and confirm
      the render job does not silently skip the cached-result chunks. Note that
      `build.yml` is the journal-managed builder (do not modify it); this is a
      documentation change in the paper, not a workflow change. (verified:
      `build.yml` has no test step; `test.yml` runs `julia-runtest`)

### Claims and framing (major)

- [x] Substantiate the Local-IRLS strategy-collapse claim (@fig-irls-comparison,
      @sec-irls). The text says the three strategies "coincide ... numerically
      (verified in unit tests)" (`intercepts.qmd:1524`) and again that the
      collapse "is verified numerically in the unit tests rather than the"
      figure (`:1546`), but neither is checkable from the manuscript. Add a
      panel / inset showing the strategies coincide on the inner quadratic, or
      report the numerical agreement tolerance and point to the specific test.
      (verified: claim is asserted, not shown)
- [x] Reconcile the per-run suboptimality reference across the multinomial
      figures. @fig-multinomial-imbalance (`:1204`) and @fig-multinomial-sweep
      (`:1303`) measure suboptimality against each run's own lowest primal,
      which can understate a stalling gradient run --- exactly the risk
      @fig-real-multinomial-shuttle already guards against by referencing the
      minimum primal across the three runs (`:1353`, with a caption noting the
      stall "is not understated by its own run-local minimum"). Propagate the
      global / shared reference to the imbalance and sweep figures, or carry the
      shuttle-style caveat into their captions. (verified: imbalance + sweep use
      per-run; shuttle uses global)
- [ ] Reword the rate-gap bullet in the introduction's contributions list. The
      "A Unifying Schur-Complement Analysis" item calls
      $T_{\mathrm{G}}/T_{\mathrm{N}} \sim L_0/H_{00}$ a "leading-order
      iteration-complexity ratio" (`intercepts.qmd:183`), but @sec-imbalance-reg
      frames the same quantity as scaling and monotonicity only, with a 2--4x
      slack between bound and empirics. Distinct from the done remark-relabel of
      @prp-rate-gap (that was the structural label); this is the intro framing,
      where a reader of the contributions alone should see a *scaling*
      prediction, not a complexity result. (verified: bullet still says
      "iteration-complexity ratio")

### Clarity and presentation (minor)

- [ ] Distinguish measured from source-read placements in @tbl-classification.
      ncvreg and Lasso.jl appear in the table (`intercepts.qmd:1482`--`:1483`)
      and in "correctly orders every package in our survey" (`:191`), but
      neither is benchmarked --- the production figures cover glmnet, biglasso,
      adelie, and skglm, with LIBLINEAR / BlitzL1 already openly flagged as
      source-read. Either add ncvreg / Lasso.jl to a figure or soften the "every
      package" claim to separate measured from classified-from-source
      placements. (verified: ncvreg / Lasso.jl absent from `experiments/`)
- [x] Add a numerical-stability note for the logistic loss. `src/loss.jl:51`
      computes `sum(log1p.(exp.(η)) .- η .* y)` with no overflow-safe
      reformulation, so large $\eta$ overflows `exp` to `Inf`. The Armijo guards
      keep this off the hot path in practice, but given the paper's
      saturated-logistic and extreme-rate emphasis a one-line comment (or a
      `log1pexp`-style guard) is worth it. (verified)
- [ ] Decide on the main-text / supplement `-full` pairing. A cold referee read
      @fig-multinomial-sweep vs @fig-multinomial-sweep-full and
      @fig-warm-start(-mechanism) vs their `-full` twins as near-duplicate. The
      compact-main / full-supplement split is the intended structure (the
      2026-05-20 pass routed the dense grids to the supplement on purpose), so
      this is likely a signposting fix --- a one-clause pointer from each compact
      main figure to its supplement twin --- rather than a cut. Low priority.
      (verified: `-full` versions live in the supplement)
