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
  done, so this is now a mechanical step: - Targets (each maps 1:1 to a
  script, a `.jld2`, and a figure; all synthetic, deterministic, no external
  data). Measured warm compute (post-compile): -
  `fig-per-pass-cost`/`fig-per-pass-progress` ← `sim-per-pass-cost.jl` (6 s) -
  `fig-cold-start-poisson` ← `sim-cold-start-poisson.jl` (17 s) -
  `fig-multinomial-imbalance` ← `sim-multinomial-imbalance.jl` (22 s) -
  `fig-irls-comparison` ← `sim-irls-comparison.jl` (26 s) - `fig-mu-extreme`
  ← `sim-mu-extreme.jl` (60 s) - `fig-cold-start` ← `sim-cold-start.jl` (92 s) -
  Total added render compute: \~223 s. - Lift recipe per figure: add
  `using Intercepts` and `using DrWatson` to the `setup` chunk (drop the
  now-redundant local `using Intercepts` in `@fig-parametric`), then replace
  the chunk's `dd = JLD2.load(...); df = DataFrame(dd["results"])` preamble
  with the script's loop verbatim (its `param_dict` + `dict_list` +
  per-config `Random.seed!` + `d_exp` pushes, minus the `using`/`@save`
  lines) ending in `df = DataFrame(results)`. Lifting verbatim (not
  rewriting the loop) reproduces the cached DataFrame row-for-row, so the
  existing `flatten(...)` + AlgebraOfGraphics plotting code stays untouched.
  Notes: `fig-cold-start-poisson` keeps its `try/catch`;
  `fig-irls-comparison` needs the `run_method` helper + `method_label`;
  `fig-per-pass-cost` lifts the full custom
  `cdsolver`/`inner_steps`/`relgap_after` loop and `fig-per-pass-progress`
  reuses its `df`. - Then delete each inlined `experiments/<script>.jl` +
  `results/<name>.jld2`, and re-verify the hard-coded prose numbers in the
  per-pass section (`6/4/3`, `1.30/1.50/1.59×`, pass counts `109/107/121`
  and `109/108/120`) against the recomputed figures. - Keep cached
  regardless: the genuinely heavy sweeps (`sim-mu-reg-heatmap` --- the
  @sec-imbalance-reg heatmaps, `sim-multinomial-imbalance-sweep`,
  `sim-warmstart-path`, `sim-rate-gap`), every real-data figure (network /
  LIBSVMdata or R dependency), and all production-solver CSVs --- the CI
  render job installs Julia only, so R/Python/network work can never run
  there. - Done already: removed three orphan artifacts the notebook never
  loaded (`sim-logreg-mu`, `sim-logreg-reg`, `sim-h00-heatmap`, scripts +
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

- [ ] Decide on the main-text / supplement `-full` pairing. A cold referee read
  @fig-multinomial-sweep vs @fig-multinomial-sweep-full and
  @fig-warm-start(-mechanism) vs their `-full` twins as near-duplicate. The
  compact-main / full-supplement split is the intended structure (the
  2026-05-20 pass routed the dense grids to the supplement on purpose), so
  this is likely a signposting fix --- a one-clause pointer from each
  compact main figure to its supplement twin --- rather than a cut. Low
  priority. (verified: `-full` versions live in the supplement)

- [ ] Determine whether the frozen-Hessian result extends to a general theorem
  for randomly permuted coordinate sweeps. @prp-frozen-block-rate proves the
  exact coefficient-block result, while @fig-rho-frozen verifies the full
  permuted-sweep operator through estimated random-product rates. A theorem
  would need to characterize the relevant Lyapunov exponents and recover the
  $L_0/H_{00}$ limit under explicit coupling conditions. This extension
  would strengthen the analysis, but it is not needed to explain the
  observed centering plateau.

## Referee-pass follow-ups (2026-08-21)

From a fresh Computo-referee read of the current draft. Major items first, then
editorial improvements and pre-submission checks. The normalization and
global-descent findings are substantive; the proposed rate-section overhaul is a
presentation judgment rather than a defect in the evidence.

### Major

- [x] Reconcile the normalization of $\partial_0F$ in @fig-warm-start-mechanism
  and its discussion with the averaged objective in @eq-primal-problem.
  (Resolved: divided the diagnostic by $n$ in `sim-warmstart-path.jl`,
  updated the cached vectors by the same mechanical scaling, and changed the
  reported cold-start values from the unnormalized scores to $0.20$ and
  $0.47$. The qualitative warm-start mechanism is unaffected.)

- [x] Narrow or prove the assertion after @fig-cold-start that bounded logistic
  curvature makes an undamped Newton intercept step always descend. The
  experiments establish benign behavior for the tested, zero-initialized
  configurations, not global descent from an arbitrary intercept. Bounded
  curvature alone does not imply descent. Prefer narrowing the claim; the
  recommendation of an Armijo-guarded Newton step does not depend on it.
  (Resolved: restricted the conclusion to the tested zero-initialized
  problem family and stated explicitly that bounded curvature alone does not
  imply global descent.)

### Editorial improvements

- [x] Clarify the hierarchy of the rate explanations earlier in the section:
  @eq-grad-residual describes the first-update mechanism, @eq-rate-gap is a
  deliberately non-sharp scaling heuristic, @fig-rho-centering shows that it
  misses the long-run shape, and @prp-frozen-block-rate with @fig-rho-frozen
  explains the local plateau. The manuscript already makes each
  qualification, so first try stronger signposting and compression. Move an
  intermediate diagnostic to the supplement only if the revised section
  remains too dense. (Resolved: added an explicit roadmap before the
  heuristic, reframed the centering sweep as a test of its limit, and made
  the transition to the frozen local-rate model explicit. Kept the
  diagnostics in the main text because each now has a distinct role.)

- [x] Reduce repeated conclusions across @sec-theory, @sec-results, and the
  Discussion, especially that Newton and exact overlap, that the gradient
  update stalls under imbalance, and that production solvers divide by
  curvature class. (Resolved: rewrote the opening of the Discussion around
  the practical decision rule, the evidential hierarchy, and the scope of
  the conclusion. Removed its repeated solver catalogue and repeated
  descriptions of the Newton/exact and imbalance results.)

- [x] Document the remaining exported package functions, especially
  experiment-facing helpers. Prefer removing exports for internal experiment
  infrastructure such as `simulated_experiment`, `real_experiment`, and
  `suboptimality_against_certified_optimum!` unless they are intended as a
  supported public API. (Resolved: retained the three reproduction helpers
  as the experiment-driver interface, documented them and the other
  undocumented exports, and added a test requiring documentation for every
  exported name.)

- [x] Correct the README errors "Compute journal" and "The distinction mattes,"
  and reconcile the two authors listed at the top with the single author
  under "Authors." Make the source corrections in `README.qmd`, then
  regenerate `README.md` with `task readme`. (Resolved in the README source
  and shared abstract metadata; the author list now renders both authors.)

- [ ] Add a concise figure-to-driver table or a single orchestration command to
  the README so that a referee need not reconstruct the complete
  regeneration sequence across Julia, R, and Python scripts. Low priority
  because the current README already documents the principal commands.

### Pre-submission verification

- [ ] Before submission, execute the notebook and representative
  cache-regeneration paths to confirm Computo's necessary reproducibility
  condition. The referee assessed provenance, data and cached-result
  presence, environment pinning, CI scope, experiment layout, and code and
  test quality only by inspection.
