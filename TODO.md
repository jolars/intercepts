# TODO

Gaps in the current draft of `intercepts.qmd`, in rough priority order.

## Framing and accessibility

- [x] Rewrite the abstract for a computational statistician outside
      CD/regularized-GLMs. Lead with the empirical phenomenon ("production GLM
      solvers disagree on how to update the intercept, and the disagreement
      matters under response imbalance") and defer "Schur-complement",
      "MM-IRLS", "prox-Newton" to the body. Mirror the change in the opening
      paragraph of @sec-theory's intro.
- [x] Split the Newton recommendation by loss family in the Discussion. Bare
      Newton for bounded-Hessian losses (logistic, quadratic); Newton + Armijo
      guard for unbounded-Hessian losses (Poisson at extreme rates). Currently
      the Discussion claims "requires no tuning", which contradicts the
      cold-start Poisson figure. *Resolved differently: single unconditional
      recommendation of the Armijo-guarded Newton step, so the contradiction
      dissolves rather than the recommendation splitting.*
- [x] Either add a second real multinomial dataset at a different $(K, n, p)$
      regime --- e.g. a long-tailed text classification benchmark --- or temper
      the abstract's "common regime" phrasing to "verified on a real multinomial
      benchmark." One panel does not carry the universality claim the abstract
      makes. *Added StatLog Shuttle ($K = 7$, $n = 58\,000$, $p = 9$, rarest
      class $0.02\%$) via `experiments/sim-real-multinomial-shuttle.jl` and
      `@fig-real-multinomial-shuttle`. Inverse $(K, n, p)$ regime to Yeoh, much
      stronger gradient stall.*

## Theory framing

- [ ] Downgrade @thm-profile-equiv and @cor-rate-gap. The first is a one-line
      envelope-theorem application; the second is, by the author's own remark, a
      heuristic upper bound that departs from the standard randomized-CD bound
      at three points (schedule mismatch, iterate-level $H_{00}$, one-step
      Schur). Either restate them as numbered Observations / Propositions, or
      keep the labels and rewrite the proofs to track each substitution honestly
      against the randomized-CD bound they corollarize.
- [ ] Reframe @fig-rate-gap as a post-hoc diagnostic, not a prediction. The
      caption already says $R_j^2$, $H_{00}$, $H_{jj}$, $\rho_{0j}^2$ are taken
      at the Newton-converged iterate, so the "predicted" curve is computed at
      the optimum. Either (a) relabel the figure and surrounding prose, or (b)
      evaluate @eq-rate-gap at both a cold-start surrogate ($\beta = 0$) and the
      optimum and show both.
- [ ] Spell out the substitution $|\beta_0 - \beta_0^*| = |\partial_0 F|/H_{00}$
      in the derivation of @eq-newton-error. The bound currently mixes
      Newton-residual in iterate space with $\partial_0 F$; the first-order
      substitution that bridges them should be visible.

## Reproducibility plumbing

- [ ] Move the @fig-parametric solver call out of the notebook. The chunk at
      line 1484 runs `cdsolver` three times at render time. Cost is trivial, but
      the editorial-phase contract is "load cache, not solvers". Add an
      `experiments/sim-parametric.jl` that writes the contour grid and the three
      intercept/coefficient trajectories to a JLD2; load and plot.
- [ ] Align CI Julia version with `Project.toml`. The workflow installs
      `julia: 1.12.6`; `Project.toml` declares `julia = "1.11"`; AGENTS.md says
      CI pins 1.11.7. Either bump CI to 1.11.x or widen the compat to
      `"1.11, 1.12"` and regenerate the Manifest under the chosen version.
      Document the choice in @sec-methodology.

## Smaller fixes

- [ ] @algo-cd shows the Newton branch without Armijo, while the prose says the
      Newton branch wraps Armijo. Either include the Armijo loop in the
      pseudocode (as a sub-procedure or with $\alpha$ as a parameter) or
      annotate the omission.
- [ ] Fix the Multinomial row in @tbl-glm. The link entry $\log(\mu/(1-\mu))$ is
      the binary logit. Replace with the reference-class per-component formula
      the body uses, or note explicitly that the entry shows the per-class form.
- [ ] Define the exact strategy once, and use one name for it. Currently it has
      three definitions (intro, theory, figures) and oscillates between "exact"
      and "convergence" across @sec-warmstart-path. Pick "exact" and use it
      everywhere.
- [ ] Define Bucket A / Bucket B at the opening of @sec-production-solvers. The
      mapping is implicit in @tbl-classification but the section uses the labels
      before defining them.
- [ ] Trim the "rare-class regime turns the slowdown from a corner case into the
      common regime" phrasing. It appears verbatim in the intro, the multinomial
      extension, and the Discussion.
- [ ] Spell out the diverging colormap range
      ($|\log_{10}\,\mathrm{ratio}| \le 0.1$) in the @fig-mu-reg-exact axis as
      well as the caption, so the figure stands alone when printed.
