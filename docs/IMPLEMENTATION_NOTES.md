# Implementation notes

Points where the code makes the method precise, or departs from its literal statement.
Each one is also documented in the module cited.

1. **One confidence event for both bounds** (`certify.mean_bounds`). Because φ(s, 0) = 1 − φ(s, 1),
   the two mis-set masses coincide (S⁺ = S⁻), so one upper confidence bound per group at level
   1 − δ_g covers L and U together. Set `guard.split_two_sided: true` for the δ_g/2 split.
2. **Miss rule** (`certify.cell_miss`). A cell counts as missed when its true support lies outside
   the support interval spanned by the base set. Lemma 1 needs exactly this, and it never counts
   more misses than "true class not in the base set" (`guard.miss_rule: label` gives that rule).
3. **Tighter certificates at no cost.** τ_g is capped at its largest possible value, and the
   reported lower bound is never below the contribution of the pilot organisations alone
   (`guard.clip_trivial`). GUARD-pess maximises the certificate as reported.
4. **Simulators fitted on the pilot** (R0, R3) use a random part of each group's pilot for fitting
   and the rest for rectification and certification (`rq3.fit_fraction`). Otherwise the
   calibration organisations would not be exchangeable with the targets.
5. **GUARD-cons** maximises a simultaneous lower confidence bound on the welfare gain over the ED
   and keeps the ED unless that bound is positive. The default interval is a *moderated* Student-t
   interval: the sample variance is combined with the worst-case variance as a prior with
   `guard.cons_prior_obs` degrees of freedom. A plain t-interval has zero width whenever the sampled
   residuals happen to be identical. `guard.cons_bound: hoeffding_serfling` is valid for finite
   samples but very wide. Under either choice the certificate of the selected package stays valid.
6. **GUARD-count** bounds the number of target organisations whose mis-set mass exceeds an order
   statistic of the pilot's, using the exact hypergeometric law (`bounds.exceedance_tail`).
   **GUARD-pooled** shuffles the pooled calibration sample first, because the betting bound needs
   an exchangeable order.
7. **Best-of-N** is evaluated through the exact law of the selected package's rank
   (`optimize.best_of_n_pmf`). The pressure axis `d` is the square root of the exact KL divergence
   of that law from uniform. The formula log N − (N − 1)/N from Gao et al. is reported as `d_gao`,
   but it understates the pressure as N approaches the number of packages.
8. **HedgeTune** chooses N by the exact expected pilot-estimated welfare of best-of-N
   (`baselines.hedgetune_select`). Every randomised rule has its own random stream.
9. **Ties** between packages go to the one that keeps more ED proposals (`PackageSpace.argmax`).
10. **NDH** = (best value on the true-welfare curve − value at full pressure) / (range of true
    welfare over all packages) (`goodhart.ndh`).
11. **Random streams** (`utils.rep_rng`) are derived from the project seed and their purpose.
    Streams that belong to one simulator are keyed by the simulator's id, so a simulator gets the
    same numbers in every configuration that lists it. A pilot in RQ3 depends only on the pilot
    fraction and the replication number, so all simulators and rules are compared on the same pilots.

## Running time

With 18 decisions, all 2^18 packages are scored exactly. One pilot replication (all rules, three
objectives, all certificates) takes about 0.3 s on one laptop core, so the default `rq3` takes about
40 minutes on one core. `--set project.n_jobs=8` runs the replications of `rq3`, `synthetic` and
`sensitivity` in parallel. Each replication has its own random stream, so the output is the same for
any number of workers.

## Tests

Beyond ordinary unit tests, the suite checks the properties the method relies on:

- the per-organisation sandwich of Lemma 1 on every package;
- that the bounds hold for all packages whenever the certified mass covers the truth;
- Monte Carlo coverage of the confidence bounds and of the certificate;
- that the rectified estimate is exactly unbiased over all possible pilots;
- that no decision rule or certificate changes when the unobserved stances change;
- the numbers in the two-group toy example.

`tests/test_review_regressions.py` keeps fixed the defects an independent review found in the
statistical core. Tests that need the raw dataset are skipped when it is absent.
