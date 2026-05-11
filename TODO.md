# TODO

Gaps in the current draft of `intercepts.qmd`, in rough priority order.

## The theory section overpromises against what it proves

- [x] **Reframe "Theoretical characterization" in the contributions list.**
      Done: contribution #2 is now titled "A unifying Schur-complement
      analysis" and describes the lemmas as short Taylor / envelope-theorem
      identities tying the slow regime to the Lipschitz mismatch
      $H_{00} \ll L_0$. The genuine quantitative result (rate corollary
      `@cor-rate-gap`) is retained as the load-bearing claim.

- [x] **Quantitative rate gap.** Done via the new `#### Quantitative rate
      gap` subsection in §Theory: corollary `@cor-rate-gap` decomposes
      Wright's randomized-CD bound by coordinate and reduces to
      $T_{\mathrm{G}}/T_{\mathrm{N}} \sim L_0/H_{00}(\hat\eta)$ in the
      imbalance regime, anchored by `@fig-rate-gap` showing empirical
      pass-count ratio scaling with $L_0/H_{00}$ across a $\mu_0$ sweep.
      Note for future work: the Schur factor $\bar\rho^2$ is small for
      standardized designs, so the corollary's headline scaling is the
      per-coordinate Lipschitz mismatch, not the Schur correction. A
      tighter rate analysis (linear convergence under restricted strong
      convexity) would predict the empirical constant more precisely;
      currently the prediction matches scaling within a factor ~2--3.

- [x] **Be explicit about the locality of the Newton-strategy results.**
      Done: added a sentence after the cold-start discussion of
      @cor-newton-coupling stating that both results are *local* Taylor
      statements about a single Newton step, that no global convergence
      proof for CD-with-Newton-intercept is given, and that empirical
      evidence in §Numerical experiments stands in for that.

## The Results section is too thin to support the Theory claims

- [ ] **Write narrative into §Results.** `fig-simulated` and `fig-real-logreg`
      currently appear with essentially no surrounding prose --- no caption
      for `fig-simulated`, no tie-back to specific lemmas. Each figure should
      open by naming the prediction it tests and close by reporting whether
      that prediction held.

- [ ] **Sweep production-solver experiments across imbalance and
      regularization.** The "$6\times$" (glmnet) and "$10\times$" (biglasso)
      numbers come from a single configuration ($\mu_0 = 0.99$, $n=500$,
      $p=1000$, $s=10$, one $\lambda$). The framework predicts scaling with
      $H_{00}/L_0$; a 2D heat-map of slowdown vs $(\mu_0,\ \lambda/\lambda_{\max})$
      would verify the prediction rather than illustrate it once.

- [ ] **Promote the cyclic-vs-permuted distinction in `fig-first-example`
      from a footnote to a figure or paragraph.** The most striking opening
      result --- the convergence strategy failing entirely on w1a --- relies
      on a footnoted caveat that permuted CD restores convergence. Either
      the footnote becomes the figure, or the opening claim gets softened.
      Currently the headline overstates the generality.

- [ ] **Harden the multinomial bare-Newton-block claim.** The text claims
      indistinguishability of bare/damped/convergence Newton "down to
      $\bar p_K = 0.005$ and feature amplitudes up to four times what the
      figure displays" but only shows one configuration. Add a sweep figure
      so the "structural rather than tuning" claim is visible.

- [ ] **Add a warm-started regularization-path experiment.** Real users solve
      along a $\lambda$ grid, where each new $\lambda$ starts with a small
      $|\partial_0 F|$ — exactly the regime where strategy differences may
      compress. Whether the strategy ranking survives warm starts is not
      currently addressed.

- [ ] **Per-pass cost decomposition for the convergence strategy.**
      @lem-convergence-cost claims $\Omega(k_0 - 1)$ wasted inner work per
      pass. Current figures are wall-clock only; an inner-iteration-count
      vs outer-pass-progress breakdown would make the "wasted work" claim
      concrete instead of asymptotic.

## Framing / structural

- [x] **Resolve the three-vs-four strategies framing.** Done by treating
      damped Newton as a safeguarded subcase of the Newton strategy
      throughout. The intro listing now folds the damped variant into the
      Newton bullet; the contributions list calls it an "Armijo-backtracking
      safeguard on the Newton strategy"; the paragraph after Algorithm 1
      explicitly notes that the algorithm's Newton branch is the bare step
      and the damped variant wraps the same direction in Armijo
      backtracking. Figures continue to plot bare and damped Newton as
      distinct curves where they behave distinctly.

- [x] **Steelman the convergence strategy.** Done: added a paragraph after
      the cost-versus-progress verdict (in §Theory, just before
      @lem-newton-approx) that names three regimes where the convergence
      strategy is defensible --- warm-started path solves (small $k_0$),
      strongly coupled designs (where envelope alignment offsets some of
      the per-pass overhead), and the prox-Newton/IRLS linearization
      boundary (cross-referenced to @sec-irls). The paragraph explicitly
      scopes the recommendation against the convergence strategy to the
      cold-start, direct-CD setting outside all three.

## Smaller items

- [ ] **Methodology/reproducibility paragraph.** Datasets, $\lambda$ grids,
      convergence tolerances, timing methodology, seed counts, hardware.
      Standard Computo reviewer ask, currently absent.

- [ ] **Related work.** Beyond the intro paragraph, no engagement with prior
      treatments of intercept handling in regularized GLMs (Friedman / Hastie /
      Tibshirani, fixed-effects / centering literature, MM-algorithm tradition
      around the $1/4$ majorant).
