"""Run a registered online test stream on an explicitly configured local model service."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path


def run(args, tasks):
    from evomem.common import MODEL_NAME, digest, file_hash
    from phase1_codegen.capabilities import LocalModelCapabilities
    from phase1_codegen.engine import code_binding
    from phase1_codegen.local_service import verify_local_service, verify_model
    from phase1_codegen.program import Program
    from phase1_codegen.roles.evolver import PARAMETERS as EVOLVER_PARAMETERS
    from phase1_codegen.protocol import HOOK_VERSION
    from phase1_codegen.roles.reviewer import PARAMETERS as REVIEW_PARAMETERS
    from phase1_codegen.tokenization import Tokenizer
    from .decision import OnlineReviewer, ONLINE_REVIEW_PROMPT
    from .native import OnlineEvaluator, actor, adapter_for, verified_inputs
    from .online import REVIEWER_DIRECTED, run_stream
    from .search import ONLINE_EVOLVER_PROMPT, OnlineEvolver, repair_item

    root, domain = args.output.resolve(), args.domain
    package = Path(__file__).resolve().parent
    model_hash = digest(verify_model())
    tokenizer = Tokenizer()
    if domain == 'alfworld':
        from evomem.alma_backend import ALFWORLD_ACTION_POLICY, ALFWORLD_ACTOR_PARAMETERS
        from scripts.alma_feasibility import PIN, VENDOR
        from benchmarks.alfworld import official_panel
        from benchmarks.alfworld import panel
        actor_parameters = ALFWORLD_ACTOR_PARAMETERS
        prompt = digest([file_hash(VENDOR/'envs_archive/prompts/alfworld_prompt.py'), ALFWORLD_ACTION_POLICY])
        benchmark = dict(name='EvoMemBench CrossEp-Emb', executor_commit=PIN,
                         protocol='EvoMemory online using official panel and ALMA actor',
                         panel_loader=file_hash(Path(panel.__file__)),
                         mapping_verifier=file_hash(Path(official_panel.__file__)),
                         official_mapping=file_hash(panel.AUDIT/'crossep_emb_mappings_test.json'),
                         audited_panel=file_hash(panel.AUDIT/'crossep_emb_manifest_audit.json'),
                         native_metadata=file_hash(panel.METADATA))
    else:
        actor_parameters = actor.PARAMETERS
        prompt = actor.PROMPTS['math' if domain == 'math' else 'code']
        benchmark = verified_inputs(domain)
    binding = dict(model=MODEL_NAME, model_hash=model_hash, endpoint=actor.URL, seed=args.seed,
                   first_answer_only_for_primary_score=True, task_memory_scope='family',
                   evolution_memory_scope='dataset', policy=REVIEWER_DIRECTED,
                   actor_parameters=actor_parameters, actor_prompt=prompt,
                   evolver_parameters=EVOLVER_PARAMETERS, reviewer_parameters=REVIEW_PARAMETERS,
                   evolver_prompt=digest(ONLINE_EVOLVER_PROMPT), reviewer_prompt=digest(ONLINE_REVIEW_PROMPT),
                   software=dict(code_binding(), **{'math_apps_online/'+p.name: file_hash(p)
                                                     for p in sorted(package.glob('*.py'))}),
                   benchmark=benchmark)
    families = {}
    with ExitStack() as resources:
        service = verify_local_service(actor.URL)
        if not service['checkpoint_root_verified']:
            raise RuntimeError('The configured service must advertise the registered local checkpoint root')
        binding['reported_checkpoint'] = service['reported_checkpoint']

        def evaluator(task):
            family = task['family']
            if family not in families:
                directory = root/'native'/family
                caps = LocalModelCapabilities(root/'capabilities', domain, directory/'capabilities.jsonl',
                                              tokenizer, model_hash, actor.URL, phase='online/'+family)
                resources.callback(caps.close)
                ctx = dict(environment=domain, family=family, memory_scope='family', memory_hook_version=HOOK_VERSION)
                family_tasks = [t for t in tasks if t['family'] == family]
                if domain == 'alfworld':
                    from .alfworld import ALFWorldOnlineEvaluator
                    families[family] = ALFWorldOnlineEvaluator(directory, family_tasks, tokenizer, model_hash,
                                                              ctx, capabilities=caps, base_url=actor.URL)
                    resources.callback(families[family].close)
                else:
                    families[family] = OnlineEvaluator(directory, domain, adapter_for(domain), family_tasks,
                                                      tokenizer, model_hash, ctx, caps, root)
            families[family].begin_item(task)
            return families[family]

        def evaluate(*, task, source, logs, snapshot, directory):
            native = evaluator(task)
            program = Program(source)
            before = native.snapshot(program, logs, snapshot)
            first, = native(program, logs, [task], 'first', memory_snapshot=before)
            return first, before

        def repair(*, task, source, logs, snapshot, first_episode, records, directory):
            native = evaluator(task)
            with ExitStack() as roles:
                proposer = OnlineEvolver(tokenizer, directory/'evolver.jsonl', model_hash, base_url=actor.URL)
                roles.callback(proposer.close)
                reviewer = OnlineReviewer(tokenizer, directory/'reviewer.jsonl', model_hash, base_url=actor.URL)
                roles.callback(reviewer.close)
                seed = int(digest([args.seed, task['id']])[:7], 16)
                return repair_item(out=directory/'repair', tokenizer=tokenizer, proposer=proposer,
                    reviewer=reviewer, evaluator=native, task=task, parent=Program(source), logs=logs,
                    parent_snapshot=snapshot, first_episode=first_episode, records=records,
                    seed=seed, capabilities=native.caps)

        def commit(*, task, source, logs, snapshot, episode, directory):
            return evaluator(task).commit(Program(source), logs, task, episode, memory_snapshot=snapshot)

        return run_stream(root=root, domain=domain, tasks=tasks, binding=binding, evaluate=evaluate,
                          repair=repair, commit=commit, condition=args.condition)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'))
    parser.add_argument('domain', choices=('math', 'apps', 'alfworld'))
    parser.add_argument('--output', type=Path, help='New directory or the identical registered run to resume')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--per-family', type=int, help='Optional predeclared cap per family (default: all eligible)')
    parser.add_argument('--condition', choices=('cross_type_em', 'no_external_em'), default='cross_type_em')
    args = parser.parse_args()
    if args.action == 'run' and args.output is None:
        parser.error('run requires --output')
    if args.per_family is not None and args.per_family <= 0:
        parser.error('--per-family must be positive')
    from collections import Counter
    from evomem.common import digest
    from .native import load_tasks
    from .online import PROTOCOL, REVIEWER_DIRECTED
    if args.domain == 'alfworld':
        from benchmarks.alfworld.official_panel import load_panel
        from benchmarks.alfworld.panel import FAMILIES
        panel = load_panel()
        tasks = []
        for family in FAMILIES.values():
            pool = sorted((t for t in panel if t['family'] == family),
                          key=lambda t: digest([args.domain, args.seed, t['id']]))
            tasks.extend(pool[:args.per_family])
    else:
        tasks = load_tasks(args.domain, seed=args.seed, per_family=args.per_family)
    if args.action == 'plan':
        result = dict(protocol=PROTOCOL, domain=args.domain, split='test',
                      families=dict(Counter(t['family'] for t in tasks)), ordered_tasks=digest(tasks),
                      condition=args.condition, seed=args.seed, repair_policy=REVIEWER_DIRECTED)
    else:
        result = run(args, tasks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
