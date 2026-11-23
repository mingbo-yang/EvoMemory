# Paired proposal-policy search

Use `python -m rsi_paired` for the ALFWorld engineering batch. The Executor
uses the ALMA action loop. Memory programs implement `on_event(state, event,
context)`. The Evolver can rewrite the restricted code and English instructions
in its own proposal policy. The Reviewer, self-revision decision, revision
procedure, data split and adoption rule are fixed.

## What a batch measures

1. Run the learning stream once and freeze the mechanism and type-local Task
   Memory. Record the learning prefix and parent diagnosis.
2. Generate three independent primary candidates from that one snapshot. After
   all slots are closed, evaluate the parent and valid candidates on a common
   ordered panel. Keep the best, favoring the parent on a tie.
3. Let the Evolver decide whether to revise its proposal policy. If triggered,
   validate one revised policy. The old and revised policies each receive three
   new slots. All six start from the original snapshot, read the same sealed
   post-primary evidence and use matched seeds. Neither receives sibling scores.
4. Evaluate the new candidates, parent and primary winner on a second ordered
   panel. Both arms can select either common option. Adopt the new policy if its
   best option solves at least one more task than the old arm's best. Otherwise
   retain the old policy and its best mechanism.
5. Commit the winner's pre-evaluation Task Memory and append the batch history.
   Each panel uses private, online-updating memory. Persisted history can guide
   the next batch within the dataset. The CLI runs one engineering batch.

Each proposal has an initial submission and up to three pre-scoring repairs for
syntax or interface faults. Every attempt consumes its slot's resources. Invalid
slots have no native score. A completed negative result stays in the record.
Host failures are journaled and can resume within the registered attempt limit.

## Reproduce a small native run

Use the editable install in the repository README and install the ALFWorld extra.
Prepare assets explicitly and use a running local OpenAI-compatible model service.
The commands below take six official training games, two each for learning,
primary evaluation and paired evaluation. The same two task families occur in all
three panels. Selection excludes game groups and content shared with non-training
splits. Supply additional local exclusion manifests with repeated `--exclude`
arguments. The panel stores the task order, environment seeds and archive hashes.

```bash
pip install -e '.[alfworld,models,dev]'
python -m scripts.prepare_assets alfworld
python -m rsi_paired prepare-smoke \
  --games data/downloads/alfworld-tw-0.4.2.zip \
  --metadata data/downloads/json_2.1.1_json.zip \
  --out runs/panel.json
python -m scripts.register_model --model "$EVOMEMORY_MODEL_PATH" \
  --output runs/checkpoint.json
```

Create a local model config pointing to the service and checkpoint you registered.
For example, the following command writes a private config using your environment
variables. Set `MODEL_CONTEXT_TOKENS` to a length supported by your running
service. The endpoint and checkpoint are verified before learning.

```bash
python - <<'PY'
import json, os
from pathlib import Path
config = dict(base_url=os.environ['EVOMEMORY_BASE_URL'],
              model_name=os.environ['EVOMEMORY_MODEL_NAME'],
              tokenizer_path=os.environ['EVOMEMORY_MODEL_PATH'],
              context_tokens=int(os.environ['MODEL_CONTEXT_TOKENS']),
              checkpoint_manifest=str(Path('runs/checkpoint.json').resolve()))
Path('runs/local-model.json').write_text(json.dumps(config))
PY
EVOMEMORY_ENABLE_OPTIONAL_MODELS=0 TMPDIR=/tmp python -m rsi_paired smoke \
  --manifest runs/panel.json --model-config runs/local-model.json --out runs/smoke
```

The example uses the registered text model. Set the optional capability setting
to 1 only with the auxiliary model assets and multimodal runtime described in
the project configuration. The service process and GPU allocation are managed
independently of these commands.

To exercise the paired branch in an explicitly labeled engineering run, use a
separate output directory and pass `--exercise-pair`. The natural trigger
decision is still recorded. A forced exercise tests plumbing and is excluded
from evidence of autonomous triggering.

A complete batch with six distinct tasks uses at most 26 planned native episodes,
nine proposal slots and 36 mechanism submissions, plus separate model calls for
review, self-revision and optional capabilities. Independent infrastructure
attempts and costs are recorded separately. Keep the full run directory private:
it contains model messages, mechanisms, environment observations and local paths.
Re-run the same command to recover eligible unfinished work.

## Inspect or audit a run

`python -m rsi_paired.audit runs/smoke` reports the current phase, sealed slots,
learning and evaluation receipts and usage. Add `--require-complete` for a
packaging or CI gate. The report checks original-parent and policy identities,
the public evidence delivered in proposal requests, panel results and the
deployment decision against the underlying receipts.
