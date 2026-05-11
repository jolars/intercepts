# TODO

Gaps in the current draft of `intercepts.qmd`, in rough priority order.

## The theory section overpromises against what it proves

- [ ] **Reframe "Theoretical characterization" in the contributions list.** The
      Schur / $\rho_{0j}^2$ view is a *lens*; the supporting lemmas are
      one-line Taylor expansions and one envelope-theorem identity. The
      contributions list currently advertises "Three lemmas predict ... exactly
      which strategy-by-family combinations slow", which a reviewer will read
      as a quantitative result. Either weaken the framing to match what is
      proved (a unifying analysis), or supply a real rate result (next item).

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

- [ ] **Be explicit about the locality of the Newton-strategy results.**
      @lem-newton-approx and @cor-newton-coupling are local (Taylor) bounds.
      There is no global convergence proof for CD-with-Newton-intercept.
      One sentence acknowledging this would head off a reviewer ask.

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

- [ ] **Resolve the three-vs-four strategies framing.** The intro says "three
      main strategies" then bolts on damped Newton as "a fourth variant".
      Algorithm 1 omits damped Newton entirely. Pick a frame: either four
      strategies throughout, or treat damped Newton as a safeguarded subcase
      of Newton (closer to what the theory actually does).

- [ ] **Steelman the convergence strategy.** §IRLS already partly does this
      by rehabilitating it at the prox-Newton boundary, but the direct-CD
      verdict is currently a one-sided dismissal. One paragraph naming the
      regimes where convergence is defensible (warm-started path solves,
      very ill-conditioned designs, prox-Newton boundary) would sharpen the
      recommendation rather than weaken it.

## Smaller items

- [ ] **Methodology/reproducibility paragraph.** Datasets, $\lambda$ grids,
      convergence tolerances, timing methodology, seed counts, hardware.
      Standard Computo reviewer ask, currently absent.

- [ ] **Related work.** Beyond the intro paragraph, no engagement with prior
      treatments of intercept handling in regularized GLMs (Friedman / Hastie /
      Tibshirani, fixed-effects / centering literature, MM-algorithm tradition
      around the $1/4$ majorant).
