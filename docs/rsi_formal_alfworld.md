# Formal ALFWorld campaign

The 240 distinct games cover six native ALFWorld task families. Each family has
20 development games and 20 final games. Four development batches each use
12 learning games, six primary search games and six paired search games.
The remaining 24 development games form the core confirmation panel. If a
batch never scores the paired comparison, its six reserved games join that
panel for every checkpoint. The rule depends on whether a comparison ran.
A pre-freeze receipt verifies that all 120 development games were reached.
Development draws from the official training split. Final evaluation prefers
the official unseen validation split and fills any shortage from the seen
validation split. Development and final game groups are disjoint.

Each batch inherits the previous mechanism, proposal policy, family-scoped
Task Memory and evolution history. It uses the natural self-revision trigger.
Confirmation compares the initial framework and committed frameworks from
empty, separate Task Memory on identical games and seeds. A tie goes to the
earlier checkpoint. Each unscored paired comparison can add 30 confirmation
episodes across the five checkpoints, within the overall maximum of 576
episodes across development, confirmation and final evaluation, before
infrastructure retries.

The final run starts with empty Task Memory and uses a frozen mechanism and
proposal policy. The fixed Executor applies the frozen Hook, which can update
family-scoped Task Memory after a task. Reports count verified native rewards
and mechanism execution failures separately.

```bash
python -m rsi_paired prepare-formal \
  --games data/downloads/alfworld-tw-0.4.2.zip \
  --metadata data/downloads/json_2.1.1_json.zip \
  --out runs/formal-panel.json
python -m rsi_paired campaign --manifest runs/formal-panel.json \
  --model-config runs/local-model.json --out runs/formal
```

Follow [the local-service setup](rsi_paired.md). To resume the client, use the
same source, model registration and output directory. The campaign seals
`frozen.json` before final evaluation. It writes the per-family and
per-split counts to `final/summary.json` and a completion receipt to
`complete.json`. Inspect each batch with
`python -m rsi_paired.audit runs/formal/evolution-01`.
