# TODO

Gaps in the current draft of `intercepts.qmd`, in rough priority order.

## Major

- [x] **Unify the $F$ scaling.** Committed to averaged
      $F = (1/n)\sum_i f$ with $L_0 = 1/4$. Propagated through loss definitions,
      gradient/Hessian displays, every numbered proposition (including the
      $H_{00}^3$ correction in @prp-newton-approx), @algo-cd is symbolic and
      unchanged, @tbl-classification updated, and the IRLS section now uses
      $\tilde F^{(t)} = (1/(2n))\sum w_i(\cdot)^2$. Scaling-invariant ratios
      ($H_{00}/L_0 = 4\mu_0(1-\mu_0)$, etc.) are unchanged, so the figure
      results and @cor-rate-gap predictions remain valid.
- [x] **Reframe @cor-rate-gap → @eq-rate-gap-asymptotic.** Took option (c):
      `experiments/sim-rate-gap.jl` now records $R_0^2$, $R_j^2$, and the full
      @eq-rate-gap ratio (evaluated at the Newton-converged iterate on the
      standardized design); @fig-rate-gap plots empirical vs that full ratio
      instead of $L_0/H_{00}$. Corollary title and prose retightened to frame
      @eq-rate-gap-asymptotic as the $|\beta_0^\star|/\|\beta^\star\| \to \infty$
      limit; surrounding paragraphs now attribute the residual offset to
      CD-bound slack rather than to dropped subdominant terms.
- [x] **Widen the production-solver evidence past one design.** biglasso,
      adelie, skglm, and the prox-Newton pair now run on w1a and news20-3pct
      in addition to the synthetic $\mu_0 = 0.99$ problem; each figure in
      §2.2 facets by dataset. biglasso and adelie were switched to path
      mode (warm-start from $\lambda_{\max}$ down to
      $0.05\,\lambda_{\max}$) to sidestep the cold-start single-lambda
      overshoot glmnet already worked around (see new item below). skglm
      runs with `max_iter=2000` and a tol grid trimmed at $10^{-7}$ to bound
      gradient-stall wall time on news20-3pct. `X.csv` files are gitignored
      and regenerate from `experiments/sim-real-problem.jl`.
- [x] **Document the cold-start intercept overshoot in production solvers.**
      Done: `experiments/sim-cold-start-real.R` runs biglasso (Newton vs
      MM) and adelie at single $\lambda$, cold start, $\text{eps} = 10^{-1}$,
      sweeping the iterate cap from $10^1$ to $10^5$ on w1a and news20-3pct.
      Results in `results/real-solvers/cold-start-diag.csv` and surfaced in
      @sec-cold-start-diag (subsubsection of Production Solvers, with
      @tbl-cold-start-diag and a forward-pointing footnote at the path-mode
      methodology paragraph). Findings: on w1a, biglasso `Newton`
      oscillates ($\beta_0 \in [-1722, -291]$ across the sweep, primal
      stuck near $10^1$), adelie returns an empty state regardless of
      `irls_max_iters`, and biglasso `MM` converges in 2 iterations. On
      news20-3pct the catastrophic divergence does not reproduce; biglasso
      `Newton` instead exits its eps criterion prematurely (primal $0.31$
      vs path-mode optimum $0.0497$) while MM gives $0.089$ at the same
      eps. Two-dataset claim survives in a more measured form than "both
      blow up to $-1500$."
- [x] **Isolate the Armijo guard's contribution.** Done: added a new
      `UnguardedNewtonStrategy` (singleton type in `src/intercept_strategies.jl`,
      exported, dispatch symbol `:unguarded_newton`) that takes the full Newton
      step on the intercept without the Armijo loop, plus equivalence /
      cold-start tests in `test/cd.jl`. Re-ran @fig-cold-start with the
      unguarded variant as a third trajectory: on logistic at
      $\mu_0 \in \{0.5, 0.9, 0.95, 0.99\}$ the bare Newton step matches
      guarded Newton and the exact strategy pointwise across all seeds, so
      the guard is dormant in the bounded-Hessian regime. Added
      @fig-cold-start-poisson (new experiment
      `experiments/sim-cold-start-poisson.jl`, cached at
      `results/sim-cold-start-poisson.jld2`) on Poisson at
      $\mu_0 \in \{10, 30, 100, 300\}$, the unbounded-Hessian regime: bare
      Newton overshoots the cold-start intercept by an amount that grows with
      $\mu_0$ (pass-one primal spike from $500$ up to $3.3\times 10^6$ at
      $\mu_0 = 300$), the cdsolver's per-coordinate Armijo absorbs it over
      subsequent passes so the final primal still matches, but the
      pass-one relative gap sits several orders of magnitude above the
      guarded trajectory. Refit captions and the Discussion recommendation
      accordingly: the guard is free insurance on bounded-Hessian losses and
      load-bearing on unbounded-Hessian losses.
- [x] **Trim the formal apparatus.** Hybrid landing: kept @thm-profile-equiv
      as a labeled theorem (it's the substantive bridge result the rest of
      the paper anchors to) with its proof shortened to a one-liner naming
      the envelope theorem; collapsed @prp-exact-cost, @prp-newton-approx,
      @prp-newton-coupling, and @prp-gradient-partial to in-text remarks
      anchored to the canonical identity. Preserved @eq-intercept-drift,
      @eq-newton-error, @eq-newton-coupling, @eq-newton-good-enough,
      @eq-grad-residual as the citable handles for the four collapsed
      propositions. Rewrote ~45 prop-level cross-references to use the
      surviving equation labels or named-identity phrasing ("within-pass
      drift", "Newton-coupling bias", "Schur-coupling residual"); the 5
      @thm-profile-equiv cites remain unchanged. No code/result changes;
      paper renders without unresolved references.

## Minor

