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

- [x] Downgrade @thm-profile-equiv and @cor-rate-gap. *Demoted both to
      Propositions (`@prp-profile-equiv`, `@prp-rate-gap`). For the rate gap,
      replaced the `proof` environment with a running-prose derivation that
      flags the three heuristic substitutions inline (schedule, iterate-level
      $H_{00}$, one-step Schur) and shrunk the trailing remark.*
- [x] Reframe @fig-rate-gap as a post-hoc diagnostic, not a prediction. *Took
      option (b). `experiments/sim-rate-gap.jl` now also evaluates @eq-rate-gap
      at $\eta_i^{(0)} = \mathrm{logit}(\mu_0)$ (the cold-start proxy of
      @sec-imbalance-reg). The figure plots three curves: cold-start prediction,
      iterate-level diagnostic, empirical. They cross near $\mu_0 \approx 0.97$;
      neither bound uniformly tighter. Caption and surrounding prose rewritten
      to label each curve's evaluation point and flag the heuristic-substitution
      slack.*
- [x] Spell out the substitution $|\beta_0 - \beta_0^*| = |\partial_0 F|/H_{00}$
      in the derivation of @eq-newton-error. The bound currently mixes
      Newton-residual in iterate space with $\partial_0 F$; the first-order
      substitution that bridges them should be visible. *Replaced the one-line
      lead-in with a three-step chain: Taylor identity for $\partial_0 F$ at
      $\beta_0^*$ (`@eq-intercept-taylor`), divide-by-$H_{00}$ recovers the
      iterate-space Newton bound (`@eq-newton-iterate`), then the first-order
      reading of the same identity supplies $\beta\_0 - \beta\_0^* = \partial\_0
      F/H\_{00} + O(\cdot)$ and substitutes into @eq-newton-iterate to give
      @eq-newton-error.\*

## Reproducibility plumbing

- [ ] Move the @fig-parametric solver call out of the notebook. The chunk at
      line 1484 runs `cdsolver` three times at render time. Cost is trivial, but
      the editorial-phase contract is "load cache, not solvers". Add an
      `experiments/sim-parametric.jl` that writes the contour grid and the three
      intercept/coefficient trajectories to a JLD2; load and plot.
- [x] Align CI Julia version with `Project.toml`. The workflow installs
      `julia: 1.12.6`; `Project.toml` declares `julia = "1.11"`; AGENTS.md says
      CI pins 1.11.7. Either bump CI to 1.11.x or widen the compat to
      `"1.11, 1.12"` and regenerate the Manifest under the chosen version.
      Document the choice in @sec-methodology. *Aligned upward: bumped
      `Project.toml` compat to `julia = "1.12"` to match the already-pinned CI
      (`1.12.6`) and the `Manifest.toml` (`julia_version = "1.12.6"`). Updated
      AGENTS.md's "Project targets Julia 1.11 (CI pins 1.11.7)" note and the
      @sec-methodology sentence to spell out 1.12 with the 1.12.6 CI pin.*

## Smaller fixes

- [x] @algo-cd shows the Newton branch without Armijo, while the prose says the
      Newton branch wraps Armijo. Either include the Armijo loop in the
      pseudocode (as a sub-procedure or with $\alpha$ as a parameter) or
      annotate the omission. *Inlined the Armijo loop into the Newton branch:
      compute $\delta$ from the Newton direction, initialize $\alpha = 1$,
      shrink $\alpha \gets \alpha/2$ while the sufficient-decrease condition
      $F(\beta\_0 + \alpha\delta) > F(\beta\_0) + c\,\alpha\,\delta\,\partial\_0
      F$ fails, then step $\beta_0 \gets \beta_0 + \alpha\delta$. Updated the
      trailing prose to identify $c \in (0,1)$ (we use $10^{-4}$). Gradient and
      Exact branches unchanged --- they are not Armijo-guarded in the
      implementation.*
- [x] Fix the Multinomial row in @tbl-glm. The link entry $\log(\mu/(1-\mu))$ is
      the binary logit. Replace with the reference-class per-component formula
      the body uses, or note explicitly that the entry shows the per-class form.
      *Replaced the link cell with $\log(\mu/(1 - \sum_{j=1}^{m-1}\mu_j))$ ---
      the reference-class form consistent with the loss and inverse-link cells.
      Extended the caption to spell out the reference-class parameterization
      ($\eta_m \equiv 0$, $\mu_m = 1 - \sum_{j<m}\mu_j$, $\eta$/$\mu$ are
      $(m-1)$-vectors with element-wise $\log$/$\exp$) so the row reads
      unambiguously.*
- [x] Define the exact strategy once, and use one name for it. Currently it has
      three definitions (intro, theory, figures) and oscillates between "exact"
      and "convergence" across @sec-warmstart-path. Pick "exact" and use it
      everywhere. *Promoted the intro "Exact Strategy" bullet (cleaned up the
      awkward "in the intercept convergence" phrasing) to the canonical
      definition; trimmed the theory enumeration item 3 to just name "the* exact
      strategy*" without redefining; and swapped every remaining strategy-named
      "convergence" → "exact" (the contribution bullet, the @sec-theory lead
      paragraph, the @prp-profile-equiv statement, and the @sec-warmstart-path
      warm-start commentary). Descriptive uses of "convergence" (loop,
      tolerance, speed) left alone.*
- [ ] Define Bucket A / Bucket B at the opening of @sec-production-solvers. The
      mapping is implicit in @tbl-classification but the section uses the labels
      before defining them.
- [ ] Trim the "rare-class regime turns the slowdown from a corner case into the
      common regime" phrasing. It appears verbatim in the intro, the multinomial
      extension, and the Discussion.
- [ ] Spell out the diverging colormap range
      ($|\log_{10}\,\mathrm{ratio}| \le 0.1$) in the @fig-mu-reg-exact axis as
      well as the caption, so the figure stands alone when printed.
