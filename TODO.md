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
- [ ] **Reframe @cor-rate-gap → @eq-rate-gap-asymptotic.** The reduction to
      $L_0/H_{00}$ requires the intercept's $R_0^2$ term to dominate
      $\sum_{j\ge 1}\tilde H_{jj} R_j^2$, which fails at $\mu_0 = 0.5$ where
      $|\mathrm{logit}(0.5)| = 0$. Yet @fig-rate-gap compares the leading-order
      formula to empirical across the full $\mu_0$ range and reads the
      $\sim 3\times$ gap at $\mu_0 = 0.5$ as a constant offset. Options: (a)
      reframe the asymptotic as a bound approached only when
      $|\beta_0^\star| \gg \|\beta^\star\|$ and drop the small-$\mu_0$
      comparison; (b) report the magnitude of the dropped $\sum_j R_j^2$ terms
      per cell so the reader sees where the limit is valid; (c) plot the full
      @eq-rate-gap expression instead of the asymptotic.
- [ ] **Widen the production-solver evidence past one design.**
      @fig-real-biglasso, @fig-real-skglm, @fig-real-proxnewton, and
      @fig-real-adelie all run on the single $\mu_0 = 0.99$ shared problem; only
      @fig-real-glmnet (w1a) and @fig-skglm-controlled (heatmap) span a second
      design. Either extend biglasso, adelie, and the prox-Newton pair across a
      coarse $(\mu_0, \lambda)$ grid (e.g. 3 $\mu_0 \times$ 2 $\lambda$), repeat
      them on w1a or news20-3pct, or scope the discussion claim explicitly to
      single-problem evidence.
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
