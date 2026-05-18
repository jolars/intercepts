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
- [ ] **Document the cold-start intercept overshoot in production solvers.**
      Diagnostic on w1a (biglasso `alg.logistic = "Newton"`, single-lambda,
      `eps = 1e-1`, `max.iter` up to $10^5$) shows biglasso's unguarded
      Newton intercept overshoots from $\beta_0 = 0$ to $\approx -1500$
      (vs optimum $\approx -3.5$) and oscillates indefinitely; the
      conservative MM majorant ($w \equiv 1/4$) converges in 2 iterations
      on the same problem. adelie's outer-IRLS step shows the analogous
      failure. Both are real-world demonstrations of the unguarded-Newton
      failure mode that motivates the Armijo guard, and both are why the
      production-solver figures use path mode. Decide how to surface:
      footnote at the path-mode mention citing the diagnostic, a small
      appendix table ($\beta_0^{\text{final}}$, primal, iter at varying
      `max.iter`), or fold into @fig-cold-start as a production-solver
      instance. Closely related to "Isolate the Armijo guard's
      contribution" below; might subsume it.
- [ ] **Isolate the Armijo guard's contribution.** @fig-cold-start is consistent
      with both "guard saved us at cold start" and "guard never fires." Add an
      unguarded-Newton baseline on the most adversarial $\mu_0$ row, or
      construct a Poisson regime with large $f'''$ where the guard fires
      nontrivially. If the guard never fires anywhere, simplify the
      recommendation to bare Newton.
- [ ] **Trim the formal apparatus.** @thm-profile-equiv is the envelope theorem;
      @prp-exact-cost, @prp-newton-approx, @prp-newton-coupling, and
      @prp-gradient-partial are one-step Taylor identities or substitutions.
      Either collapse them into in-text remarks anchored to the canonical
      identity, or keep them numbered but shorten each proof to one line naming
      the identity ("envelope theorem at $\partial_0 F = 0$"). Either way the
      contribution does not depend on the scaffolding.

## Minor

- [ ] Unify naming: "Exact Strategy" (introduction, propositions) vs
      "convergence strategy" (@sec-irls, @sec-warmstart-path, Discussion). Pick
      one and replace throughout.
- [ ] Fix the multinomial loss in @tbl-glm: the $\log(1 + \sum_j e^{\eta_j})$
      normaliser is repeated $(m-1)$ times as written. Move it outside the outer
      sum or use $\log(1 + \sum_j e^{\eta_j}) - \sum_k y_k \eta_k$.
- [ ] Reconcile `DampedNewtonStrategy`: line 987 calls it a deprecated alias,
      but the source still exports it. Either commit to the deprecation in the
      source or stop calling it deprecated in the prose.
- [ ] Reconcile the warm-start carve-out (@sec-warmstart-path: "convergence
      strategy becomes a defensible alternative to bare Newton whenever a path
      solve is in fact used") with the Discussion's flat "bare Newton in direct
      CD" recommendation. Either fold the warm-start regime into the Discussion
      or note it as a scoped exception.
- [ ] Name the deterministic procedure for the news20-3pct subsample in the
      @tbl-real-logreg-problems caption (seed, sampling order); currently only
      `experiments/sim-real-logreg.jl` has it.
- [ ] Pin the Computo Quarto extension version in the README's `quarto add`
      instruction, matching the devenv.nix pinning story for Julia, R, and
      Python.
- [ ] Add CPU model and memory to the hardware line in @sec-methodology so a
      referee replicating @fig-real-proxnewton's "tens of milliseconds" can tell
      hardware mismatch from code-path mismatch.
- [ ] Fix the @fig-real-skglm caption citations: the slow ramp is the signature
      of @prp-gradient-partial and @cor-rate-gap, not @prp-newton-coupling.
- [ ] Add a one-line README inside `results/skglm-controlled/` describing what
      `index.csv` and `results.csv` each contain.
