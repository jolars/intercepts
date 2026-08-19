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
  caption both use $H_{00}/L_0 = 4\mu_0(1 - \mu_0)$; the slowdown ordering
  at line 1712 uses the reciprocal $L_0/H_{00}$), so this is signposting for
  a reader checking the arithmetic, not a bug. (verified: not a
  header/caption mismatch) (resolved: kept the slowdown framing---it is the
  ordering rationale and is used in both the prose and the @fig-real-logreg
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
  gradient-strategy no-op ($L_0 = \infty$). They serve the theory→experiment
  arc, sit in different sections, and already cross-reference rather than
  repeat---@fig-real-poisson cites @fig-cold-start-poisson to explain the
  guarded/unguarded curve coincidence. Folding would relocate a synthetic
  example into the empirical section. Kept all three Poisson figures
  (@fig-real-poisson-production is the production-solver landscape, a
  distinct point).)

Not actionable from this pass: the review's "README structure boilerplate" point
does not apply --- `README.qmd:31` already uses `<experiments/>` and the
capitalized `Intercepts` module, and `README.qmd:56` already pins Julia 1.11.9.

## Referee-pass follow-ups (2026-05-21)

From another fresh Computo-referee read. Items tagged `(verified)` were checked
against the working tree; the rest are framing or presentation judgments to
weigh. Two of this pass's points are already tracked above and not repeated: the
Zenodo deposit (open major item under the 2026-05-20 reproducibility section),
and the abstract/intro nontechnical rewrite (done) --- the referee re-flagged
the notation in the contributions list, but keeping the Schur-complement naming
there was a conscious call recorded above.

### Reproducibility (major --- the necessary condition)

- [x] Document the CI split. Decided this is repo infrastructure, not scientific
  methodology, so it lives in `README.qmd` (a new "Continuous integration"
  subsection under Reproducibility) rather than in `@sec-methodology` ---
  keeps the paper's methodology uncluttered, and a referee checking
  reproducibility reads the repo, not the prose. `build.yml` now delegates
  to Computo's reusable workflows (`global-env.yml` builds the env,
  `publish-render.yml` renders + deploys Pages) and never runs the test
  suite; `test.yml` runs `Pkg.test()` on push / PR and guarantees `src/`.
  Both pin Julia 1.11 to match the manifest. `build.yml` stays the
  journal-managed builder (not modified).

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
- [x] Reword the rate-gap bullet in the introduction's contributions list. The
  "A Unifying Schur-Complement Analysis" item called
  $T_{\mathrm{G}}/T_{\mathrm{N}} \sim L_0/H_{00}$ a "leading-order
  iteration-complexity ratio" (`intercepts.qmd:183`), contradicting the body
  (`intercepts.qmd:829`), which frames the same quantity as a ratio of upper
  bounds whose scaling survives, not a complexity result. Now reads
  "predicts the leading-order scaling ... of the gradient strategy's
  slowdown ... a scaling the experiments confirm up to a constant factor."

### Clarity and presentation (minor)

- [x] Distinguish measured from source-read placements in @tbl-classification.
  ncvreg and Lasso.jl appeared in the table but are never benchmarked (no
  `experiments/` script), and their rows duplicated the glmnet-`Newton` /
  biglasso-`Newton` local-IRLS rows verbatim. Dropped both rows from the
  table (every remaining row is now a measured solver: glmnet, biglasso,
  skglm, adelie, LIBLINEAR, BlitzL1), kept their classification in the
  survey prose (`:32`, `:1468`), reworded the contributions bullet so "every
  package in our survey" splits the six benchmarked from ncvreg / Lasso.jl
  placed from source-reading, and replaced the caption's fragile "rows three
  to six" with "the MM-IRLS and local-IRLS rows". (Note: LIBLINEAR / BlitzL1
  *are* benchmarked via `sim-real-proxnewton.py` --- the original TODO
  wrongly called them source-read.)
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
  this is likely a signposting fix --- a one-clause pointer from each
  compact main figure to its supplement twin --- rather than a cut. Low
  priority. (verified: `-full` versions live in the supplement)

## Referee-pass follow-ups (2026-05-21, second read)

From another fresh Computo-referee read. Items tagged `(verified)` were checked
against the working tree. One major item from this read --- a stale Julia
version in `@sec-methodology` --- was fixed directly rather than left open.

### Reproducibility / claims (major)

- [x] Fixed: `@sec-methodology` (`intercepts.qmd:1655`) said the internal-solver
  experiments ran on "Julia 1.12 (CI pins 1.12.6, matching the version under
  which `Manifest.toml` was generated)", but every other artifact says
  1.11.9 --- `README.qmd:56`, `Manifest.toml` (`julia_version = "1.11.9"`),
  `Project.toml` (`julia = "1.11"`), and `test.yml` (`version: "1.11"`). The
  "Fix version discrepancy in docs" commit fixed the README but missed the
  methodology prose. Reworded to Julia 1.11 / 1.11.9. (verified)

### Clarity and presentation (minor)

- [x] Document the cross-panel $F^\star$ convention in @fig-real-adelie. The
  adelie panel references skglm's minimum primal as $F^\star$ (`df_ref` from
  `skglm.csv`, `intercepts.qmd:2670`), while @fig-real-glmnet /
  @fig-real-biglasso / @fig-real-skglm / @fig-real-proxnewton each use their
  own panel's minimum (`p .- minimum(p)`). The caption explains adelie's
  coarser outer-IRLS criterion flooring its suboptimality but never says
  $F^\star$ comes from a different solver. State the convention in the
  caption. The shared-reference choice is defensible --- it stops a stalling
  solver from understating its own gap, the same principle the 2026-05-20
  pass propagated to @fig-real-multinomial-shuttle --- but the asymmetry
  should be visible to a reader comparing panels. (superseded and resolved by
  the shared certified-reference protocol below)

### Correctness (minor)

- [x] Tighten the @prp-profile-equiv proof. It was a one-line envelope-theorem
  invocation, and the step-size claim ($1/H_{jj}$ is "a valid descent step
  size") was asserted without naming the curvature regime. (resolved:
  expanded the proof to carry the envelope derivation explicitly --- define
  $\beta_0^\star(\beta)$, differentiate
  $\tilde F = F(\beta_0^\star, \beta)$, and show the indirect term vanishes
  by stationarity $\partial_0 F = 0$ --- and reworded the proposition's
  step-size sentence: the realized $1/H_{jj}$ step is *conservative* for
  $\tilde F$ because $1/H_{jj} \le 1/\tilde H_{jj}$, never exceeding the
  exact coordinate-minimization step of $\tilde F$'s local quadratic model
  along $e_j$, so it descends rather than overshoots --- not "barely
  sufficient".)

Not actionable from this read: the "six vs eight production solvers" point is
stale --- ncvreg / Lasso.jl were already stripped from @tbl-classification
(commit `45d27fa`); its eight rows are six packages with glmnet and biglasso
each appearing in two configurations, and the contributions bullet
(`intercepts.qmd:195`) already splits the benchmarked solvers from the two
placed by source-reading. The "theory equation density" point is a
Computo-allowed clarity judgment already weighed in the 2026-05-20 pass. The
@eq-rate-gap-asymptotic prominence point sits adjacent to the (done) rate-gap
relabeling work and is left unrecorded as a framing judgment unless raised
again.

## Centering-sweep follow-up (2026-08-12)

Raised by rewriting @fig-rho-centering from a two-point centering toggle into a
continuous $\alpha$-sweep (`experiments/sim-rho-centering.jl`). The sweep is a
strict improvement on what it replaced, but it exposes a coincidence the paper
now reports without explaining.

### Claims and framing (major)

- [x] Explain, or explicitly disown, why the centering plateau lands on
  $L_0/H_{00}$. Along the sweep the empirical
  $T_{\mathrm{G}}/T_{\mathrm{N}}$ saturates by $\bar\rho^2 \approx 0.04$ and
  then sits on the @eq-rate-gap-asymptotic limit $L_0/H_{00}$ --- within
  $5\%$ at every subsequent grid point and within $2\%$ once
  $\bar\rho^2 \gtrsim 0.09$, in both $\mu_0$ series ($3.29$ and $5.88$).
  That is far too tight to be coincidence, and a referee will ask where it
  comes from. The obvious answer is wrong: @eq-rate-gap-asymptotic is
  derived by letting the $R_0^2$ terms dominate @eq-rate-gap as
  $|\beta_0^\star|/\|\beta^\star\| \to \infty$, but the sweep reaches the
  same limit without that condition holding. The recorded intercept share
  moves non-monotonically from $0.045$ to $1.06$ across the saturated cells
  while the ratio does not respond, and the plateau is as tight at
  $\alpha = 0.3$, $\mu_0 = 0.5$ (share $0.045$) as where the share is twenty
  times larger. So the limit is reached through some other route. Options,
  in rough order of value: (a) derive the plateau directly --- the natural
  candidate is that the gradient strategy advances the intercept by a fixed
  fraction $H_{00}/L_0$ of the Newton step every pass, so once the intercept
  error is re-injected each pass by coupled coefficient updates the
  pass-count ratio is pinned at $L_0/H_{00}$ regardless of where
  $\beta_0^\star$ sits; (b) test that reading by sweeping a second design
  knob (e.g. $\rho$, or $s$) and checking whether the plateau tracks
  $L_0/H_{00}$ there too; (c) leave it as a reported empirical regularity,
  the current state of the prose, which says plainly that we do not close
  the gap. Doing (a) or (b) would turn the figure's headline from "the
  leading-order factor gets only the sign right" into a positive result.
  (verified against `results/rho-centering.jld2`, which records
  `intercept_share`, `asymptotic_ratio`, and per-cell `H00_over_L0`; the
  latter is exactly single-valued per $\mu_0$, so the sweep does isolate
  $\bar\rho^2$; resolved by @prp-frozen-block-rate and @fig-rho-frozen,
  which derive the strongly coupled exact-block limit and verify that the
  plateau tracks $L_0/H_{00}$ under additional $\rho$, sparsity, and $\mu_0$
  settings)

- [ ] Determine whether the frozen-Hessian result extends to a general theorem
  for randomly permuted coordinate sweeps. @prp-frozen-block-rate proves the
  exact coefficient-block result, while @fig-rho-frozen verifies the full
  permuted-sweep operator through estimated random-product rates. A theorem
  would need to characterize the relevant Lyapunov exponents and recover the
  $L_0/H_{00}$ limit under explicit coupling conditions. This extension
  would strengthen the analysis, but it is not needed to explain the
  observed centering plateau.

## Referee-pass follow-ups (2026-08-19)

From a fresh Computo-referee read. Major items first. This pass reopens two
points that earlier edits narrowed but did not fully settle: the descent claim
in @prp-profile-equiv and the production-panel definition of $F^\star$.

### Correctness and classification (major)

- [x] Rework the descent conclusion in @prp-profile-equiv. The current proof
  establishes the envelope-gradient identity and compares the two *local
  quadratic models*, but $1/H_{jj} \leq 1/\widetilde H_{jj}$ at the current
  point does not by itself make $1/H_{jj}$ a descent step for an arbitrary
  twice-differentiable convex profiled objective. Either (a) restrict the
  proposition to the gradient identity and local-curvature comparison, or (b)
  add an explicit coordinate-smoothness / curvature-bound assumption and
  prove finite-step descent under it. Then audit the downstream wording
  around @eq-schur-diag, @eq-intercept-drift, and every appeal to
  @prp-profile-equiv so none promotes a local quadratic statement into a
  global descent guarantee. (resolved: restricted the proposition to the
  envelope-gradient identity and the ordering of the two local
  quadratic-model steps; explicitly stated that this ordering gives no
  finite-step descent guarantee without a coordinate-curvature bound; and
  narrowed every downstream appeal to a gradient identity or a heuristic
  local-model comparison.)
- [x] Tighten the production-solver taxonomy in @tbl-classification and
  @sec-irls. For each IRLS / prox-Newton implementation, name whether the
  intercept is conditionally minimized for the original GLM loss or only for
  the frozen quadratic surrogate. In particular, verify LIBLINEAR's
  `newGLMNET` update against its algorithm or source, and withdraw or
  qualify the claim that it realizes the paper's exact original-loss
  strategy if it only converges on the quadratic subproblem. (resolved:
  verified `solve_l1r_lr` at LIBLINEAR commit `491c9f1`; its unregularized
  bias receives `z = -G/H` inside the frozen QP, and no original-loss
  intercept loop follows. Renamed the table column, labeled every
  IRLS/prox-Newton row by the objective it conditionally minimizes, and
  withdrew the downstream claim that LIBLINEAR realizes the exact
  original-loss strategy. BlitzL1 retains that classification because its
  post-step Newton loop operates on the original loss.)

### Evaluation and reproducibility (major)

- [x] Define one explicit $F^\star$ protocol for all production-solver panels in
  @sec-production-solvers. Inventory the current references used by
  @fig-real-glmnet, @fig-real-biglasso, @fig-real-adelie, @fig-real-skglm,
  @fig-real-proxnewton, and @fig-real-poisson-production; decide whether
  each figure is a within-solver comparison or a cross-solver comparison;
  and use a common, independently justified reference whenever curves are
  compared across implementations. State the convention in the methods text
  and captions. This supersedes the narrower open note above that asks only
  for an adelie-caption disclosure. (resolved: added a separate guarded-Newton
  CD reference solve for every logistic and Poisson problem, required a
  relative primal--dual gap below $10^{-8}$, cached both bounds in
  `results/production-references.csv`, and made every production panel join the
  same per-problem reference. The methods text and captions now state the
  convention.)
- [x] Make the pinned R and Python environments obvious to a cold referee. Check
  that `devenv.nix` and `devenv.lock` pin every package and external solver
  needed by the production drivers, including both skglm revisions used by
  @fig-skglm-controlled. Add a compact language-to-environment mapping to
  the README and explain how the lock file reconstructs each toolchain. If
  devenv does not fully pin one of them, add the missing pin. Run each
  documented environment entry point before submission. This is separate
  from the open Zenodo task: the reviewer found the cached outputs and
  drivers but could not infer the non-Julia dependency provenance from the
  conventional environment files. (resolved: documented the language-to-
  environment mapping and lock-file reconstruction in the README; pinned both
  controlled-comparison skglm revisions as content-addressed devenv inputs;
  changed the runner to consume those immutable sources; and smoke-tested the
  R, Python, Julia, and Quarto entry points.)

### Claims and presentation (minor)

- [x] Replace “Newton curvature stays well-scaled however imbalanced the data
  are” in the abstract / metadata and any parallel Discussion wording. Under
  extreme imbalance, $H_{00}$ itself can approach zero; the supported claim
  is that dividing by the local curvature gives an adaptive update ratio.
  Search for equivalent claims rather than fixing only the known sentence.
- [ ] Reduce main-text repetition among @fig-mu-extreme, @fig-rate-gap,
  @fig-mu-reg-gradient, and @fig-warm-start. Write down the distinct claim
  each figure establishes, retain the minimum panels needed for those
  claims, and move diagnostic or robustness panels to the supplement.
  Coordinate this with the existing open `-full` signposting task instead of
  creating another compact/full pair.
- [ ] Fix the repository-facing copy, then regenerate `README.md` with
  `task readme`: change “mattes” in `_quarto.yml` (which flows into the
  README), change “Compute journal” to “Computo journal” in `README.qmd`,
  and compare the rendered README author list with `_quarto.yml` so every
  manuscript author is represented in the intended order.
