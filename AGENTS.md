# Contributor instructions

- Make code and documentation changes in this repository. Do not modify model
  weights, third-party benchmark data, other projects, or external services.
- Keep Executor, Reviewer, Evolver, and the fixed Evolution Memory construction
  and retrieval responsibilities separate. Do not change prompts or acceptance
  rules as part of packaging or portability fixes.
- Preserve failed, negative, neutral, and invalid attempts with their actual costs.
  A working pipeline does not demonstrate an effectiveness gain.
- Never commit credentials, private judging data, model files, execution logs,
  benchmark downloads, or machine-specific process/service registrations.
- Generated mechanism code must remain behind the validated event contract,
  resource limits, and Linux syscall isolation. Never fall back to unrestricted exec.
- Run relevant CPU tests. Real model/native runs require explicit authorization;
  they are not needed for documentation, packaging, or repository maintenance.
- Keep test fixtures explicitly synthetic and independent of remote services.
