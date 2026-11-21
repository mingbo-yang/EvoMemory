# Contributing

Use a focused branch, explain the observed problem and changed behavior, and
run the relevant CPU tests. Include the model/data/configuration identity when
reporting an integration issue, but remove credentials, private tests, answers,
and sensitive execution traces.

Preserve the distinction between mechanism validity, reviewer opinion, native
task outcome, and transfer effectiveness. Do not replace errors with zero
rewards or tune on held-out outcomes. Keep dataset preparation explicit.

Packaging and portability fixes must not alter model-facing prompts, benchmark
scoring, native Executor policy, or deployment criteria implicitly.
