# How to Train Your Intercept
Johan Larsson, Frederik Fabricius Bjerre
2026-10-09

### Citation

Submitted to Computo.

### Badges

[![build and
publish](https://github.com/jolars/intercepts/actions/workflows/build.yml/badge.svg)](https://github.com/jolars/intercepts/actions/workflows/build.yml)
[![reviews](https://img.shields.io/badge/review-report-blue)](https://github.com/jolars/intercepts/issues?q=is%3Aopen+is%3Aissue+label%3Areview)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/jolars/intercepts)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/jolars/intercepts)
[![DOI:10.5072/computo.0000](https://img.shields.io/badge/DOI-10.5072%2Fcomputo.0000-034E79.svg)](https://doi.org/10.5072/computo.0000)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

### Authors’ affiliations

- [Johan Larsson](https://jolars.co) (University of Copenhagen)
- Frederik Fabricius Bjerre (University of Copenhagen)

### Abstract

Coordinate descent solvers for regularized generalized linear models
(GLMs) differ in how they update the intercept. This seemingly routine
choice can substantially affect convergence when the response is
imbalanced. For direct coordinate descent on the original GLM loss, we
compare a conservative step based on worst-case curvature, a safeguarded
Newton step based on current curvature, and repeated Newton steps that
solve the intercept subproblem. The conservative update can leave much
of the interaction between the intercept and the coefficients
unresolved, producing a slowdown when that interaction controls
convergence. A single Newton step avoids this attenuation and, in our
experiments, follows nearly the same outer trajectory as exact
minimization while requiring fewer inner steps. Simulated and real-data
experiments, including comparisons across six production solvers and a
controlled intervention within skglm, support this explanation. We also
show how the same mechanism appears in quadratic-approximation methods
and extends to multinomial models with rare classes. We recommend a
safeguarded Newton update for direct coordinate descent and full
intercept optimization within frozen local quadratic surrogates.
