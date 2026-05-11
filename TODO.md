# TODO

Gaps identified in the current draft of `intercepts.qmd`, in rough priority
order.

## Structural gaps

- [ ] **Build out the Results section.** It is currently two figures with no
      narrative, while the Theory section is fully written and makes strong
      claims (gradient strategy fails under imbalance, Newton ≈ convergence to
      leading order, convergence wastes per-pass cost). The empirical section
      should systematically back each claim --- one experiment per load-bearing
      prediction, with prose tying the result back to the relevant
      lemma/corollary.

- [x] **Add experiments driving actual production solvers.** Done in the
      new `### Production solvers` subsection of Results, with three figures
      (`fig-real-glmnet`, `fig-real-biglasso`, `fig-real-skglm`) and the
      classification table `tbl-classification`. The investigation also
      surfaced corrections to the paper's framing: LIBLINEAR and BlitzL1 are
      prox-Newton, not direct-CD; only skglm and SAGA fit "Bucket 1." See
      drivers under `experiments/sim-real-{problem,glmnet,biglasso,skglm}.{jl,R,py}`.

## Theory / framing

- [x] **Convergence Rates subsection is orphaned (around line 856).** It states
      a generic $\sum_j L_j$ bound and the $L_0 = n/4$ vs local-$L_0$
      observation, but does not tie back to the $\rho_{0j}$/Schur machinery
      built earlier. Either develop a concrete rate gap (Newton vs gradient as a
      function of $\mu_0$ or $\rho_{0j}^2$) or cut. As written it weakens rather
      than strengthens.

- [ ] **Address whether the gradient-strategy failure survives backtracking line
      search.** Lemma 3.3's failure mode hinges on $L_0 = n/4$ being looser than
      $H_{00}$. A backtracking-line-search gradient method would adapt, and that
      is what many "gradient-strategy" implementations actually do. One
      paragraph (or a small experiment) addressing this closes a likely reviewer
      objection.

- [ ] **Rewrite the contributions list to advertise the actual theoretical
      contribution.** "We provide theoretical results that explain the empirical
      findings" undersells. The contribution is the $\rho_{0j}^2$ Schur
      characterization plus the three lemmas classifying strategies by how
      completely they remove the intercept--coefficient coupling.

- [ ] **Resolve the three-vs-four strategies framing.** The intro says "three
      main strategies" then bolts on damped Newton as a "fourth variant."
      Algorithm 1 omits it. Pick a frame: either four strategies throughout, or
      treat damped Newton purely as a safeguarded subcase of Newton (which is
      closer to what the theory section actually does).

## Experiments to add

- [ ] **Cyclic vs permuted CD across strategies.** The cyclic/permuted
      distinction is currently footnoted under `fig-first-example` but is
      load-bearing for the convergence-strategy failure there. Promote to a
      figure or paragraph; show how the strategy ranking shifts under permuted
      ordering.

- [ ] **Regularization-path experiments with warm starts.** Real users solve
      along a $\lambda$ grid. Whether the strategy ranking changes under
      warm-starting (where $|\partial_0 F|$ is small at the start of each new
      $\lambda$) is not discussed. Probably tightens the case for Newton
      further, but should be checked.

- [ ] **Per-pass cost decomposition for the convergence strategy.** Lemma 3.2
      claims $\Omega(k_0 - 1)$ wasted inner work per pass. The current figures
      are wall-clock only; a breakdown of inner-iteration counts vs. outer-pass
      progress would make the "wasted work" argument concrete.

- [ ] **Harden the multinomial bare-Newton-block claim.** The text claims
      indistinguishability of bare/damped/convergence Newton "down to
      $\bar p_K = 0.005$ and feature amplitudes up to four times what the figure
      displays" but only shows one configuration. Add a sweep figure (or
      appendix) so the "structural rather than tuning" claim is visible to the
      reader.

## Smaller items

- [ ] **Methodology/reproducibility section.** Datasets, $\lambda$ grids,
      convergence tolerances, timing methodology, seed counts, hardware.
      Standard reviewer ask, currently absent.

- [ ] **Related work.** Beyond the brief intro paragraph, no engagement with
      prior treatments of intercept handling in regularized GLMs
      (Friedman/Hastie/Tibshirani, fixed-effects/centering literature, etc.).

- [ ] **Steelman the convergence strategy.** The paper dismisses it but does not
      really steelman. Are there regimes (e.g. warm-started path solves, very
      ill-conditioned designs) where it would be defensible? Even a paragraph
      noting "we considered X but found Y" sharpens the recommendation.
