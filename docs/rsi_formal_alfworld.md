# Sequential ALFWorld campaign

The `rsi_original_parent_paired_v3_6x30` formal registration runs two native ALFWorld task families in order.
The supplied seed-42 panel starts with `simple` (pick and place one object)
and continues with `two` (pick two objects and place them). Each family has
six evolution rounds and its own final test. The second family starts after
the first family's final completion receipt has been verified.

## One round

1. Execute 30 previously unseen learning games once each, recording both successes
   and failures from the environment. The deployed mechanism updates its Task Memory.
2. Seal two disjoint panels of 30 games. Each calls for six historical failures,
   six historical successes and 18 fresh games. Historical status comes from the
   original learning episode, including the current batch. An official game group
   can contribute to only one panel within the round. The registered allocator
   maximizes historical quota coverage, balances scarce outcomes across the two
   panels and uses seeded ranks for selection. Fill shortfalls with the panel's
   own preregistered fresh games. Several distinct trials in a group may appear
   in the same panel.
   Candidate scores, validation-only games and the other family's tasks are
   ineligible for historical sampling. A historical game may recur in later rounds.
3. Freeze the original parent and learning memory. The Reviewer diagnoses evidence
   bound to the executed parent. Generate three primary candidates and evaluate
   eligible implementations and the parent on the first panel. Record each slot,
   including duplicates and invalid submissions. Each slot allows three
   pre-scoring syntax/interface repairs with reproducible draft-specific seeds.
   A documented candidate Hook failure on an individual game earns zero
   selection credit for that game, retains a null native score and leaves the
   candidate eligible. Infrastructure failures pause the job for recovery.
   A verified Hook error in the deployed mechanism during formal learning
   stops that learning stream. Recovery reports the same unscored receipt;
   the task and its private memory changes are never committed.
4. The Evolver makes one decision about revising its proposal policy. A malformed
   decision retains the old Evolver and the primary winner. When triggered,
   the old and revised policies each propose three fresh candidates from the
   original snapshot saved before primary search. Both see the same post-primary
   evidence. Score the two branches on the second panel. The host selects the revised
   policy only on a strict win. When self-revision is skipped the second panel
   remains registered and unscored. After a complete meta panel, a malformed
   post-Review or an invalid citation gets one correction with the same displayed
   evidence. If the correction also violates the format, persist both errors and
   a missing opinion, then select from the verified scores. Incomplete comparisons
   cannot select a winning branch from partial scores.
5. Commit the selected mechanism, proposal policy and corresponding learning
   memory. Candidate evaluations use private copies. Each round records source
   differences, memory behavior where execution histories permit comparison,
   and paired native outcomes by historical failure, historical success and
   fresh game. These describe observed behavior and task reward.

## Final test and transfer

After round six the initial framework and the round-six committed framework
each execute the same 100 ordered games with the same seeds in distinct
evaluation jobs. Both begin with empty Task Memory. Within each arm only its
own Task Memory updates. Mechanism and proposal policy are frozen. The
round-six checkpoint goes directly to the final test.

The second family restarts with the initial mechanism, initial proposal
policy and empty Task Memory and learning ledger. Only the compact
mechanism-modification experiences from the first family's six development
rounds become eligible for retrieval alongside local experiences. Its final
test outcomes stay outside that archive.

## Game identities and independence

The registration uses official ALFWorld game files, unique across the
manifest by archive member, content hash and task ID. Distinct development
instances can belong to one game group. Every final group is excluded
in full from development. Each family reserves 180 learning instances and
360 fresh validation instances before observing outcomes. A reserve is used
only when required by sampling. With the supplied seed and original archives:

| Family | Official unseen validation | Official seen validation | Isolated training games | Final total |
| --- | ---: | ---: | ---: | ---: |
| simple | 24 | 35 | 41 | 100 |
| two | 17 | 24 | 59 | 100 |

Independence is relative to this campaign. Against the prior 240-game
registration, 22 of the new simple final games and 21 of the two final
games also appeared there. The final studies measure the two named families
in this order. Each family's independent baseline is its initial framework. The final streams cover
56 and 54 underlying game groups respectively. An earlier CPU asset check
executed a single admissible action on three final instances per family without
model inference. The native v3 engineering panel uses separate game groups.

## Run and resume

Run these commands in this checkout with the pinned ALMA checkout, both
official ZIP archives and an already registered local model. The
[local-service setup](rsi_paired.md) describes asset preparation, dependencies
and the model configuration fields. Use a fresh output directory for each
distinct model or registration. Keep the registered SciPy version on resume.
With optional capabilities enabled, the auxiliary model assets and the
registered helper at `.local/multimodal-runtime/manifest.json` must be present
in the checkout that runs the job. A text-model-only deployment sets
`EVOMEMORY_ENABLE_OPTIONAL_MODELS=0` before registration and retains that setting.

```bash
python -m rsi_paired prepare-formal --families simple two --seed 42 \
  --games data/downloads/alfworld-tw-0.4.2.zip \
  --metadata data/downloads/json_2.1.1_json.zip \
  --out runs/v3_6x30_inputs/panel.json
python -m rsi_paired prepare-smoke --v3 \
  --games data/downloads/alfworld-tw-0.4.2.zip \
  --metadata data/downloads/json_2.1.1_json.zip \
  --exclude runs/v3_6x30_inputs/panel.json --out runs/v3_6x30_inputs/smoke.json
python -m rsi_paired smoke --manifest runs/v3_6x30_inputs/smoke.json \
  --model-config runs/local-model.json --out runs/v3_smoke_natural
python -m rsi_paired smoke --exercise-pair --manifest runs/v3_6x30_inputs/smoke.json \
  --model-config runs/local-model.json --out runs/v3_smoke_forced
python -m rsi_paired campaign \
  --manifest runs/v3_6x30_inputs/panel.json \
  --model-config runs/local-model.json --out runs/v3_6x30
```

Repeat the same campaign command and configuration to resume. The registration
and completed jobs are checked against their recorded inputs. Each family's
`evolution-01` through `evolution-06` directories contain
`batch/resolved_panels.json` and `candidate_differences.json`.
`frozen.json` identifies the two final frameworks and ordered test panel.
`final/summary.json` records each executed arm, split counts, paired outcomes
and costs, the success-count difference, ordered cumulative successes, recorded
steps and successes aggregated by underlying game group. Each family registers
a maximum of 390 native episode slots per evolution round and 200 frozen test
slots (2,540 in total). Report retries and any partially completed comparison
separately from committed batches. `evolution_memory.json` binds the export to round six.
Inspect any completed round with:

```bash
python -m rsi_paired.audit runs/v3_6x30/simple/evolution-01
```
