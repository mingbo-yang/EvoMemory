"""Optional, bounded observations of generated code; never mechanism advice.

Hooks execute only inside the existing isolated worker.  ``diagnose`` merely
summarizes recorded executions: it never runs a program or calls a model.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math

from .protocol import EVENTS


TRACE_BYTES = 64 * 1024
TRACE_SITES = 96
SAMPLES_PER_SITE = 2


def _wire(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _hash(value):
    return hashlib.sha256(_wire(value).encode()).hexdigest()


def _sample(value, depth=0):
    """Bounded descriptive value, explicitly marked when it is not complete."""
    kind = type(value).__name__
    if value is None or type(value) in (bool, int):
        if type(value) is int and value.bit_length() > 256:
            return dict(type=kind, bit_length=value.bit_length(), complete=False)
        return dict(type=kind, value=value, complete=True)
    if type(value) is float:
        return dict(type=kind, value=value if math.isfinite(value) else str(value),
                    complete=math.isfinite(value))
    if type(value) is str:
        return dict(type=kind, value=value[:180], length=len(value), complete=len(value) <= 180)
    if type(value) in (list, tuple, dict, set):
        out = dict(type=kind, length=len(value), complete=False)
        if depth >= 2:
            return out
        if type(value) is dict:
            items = []
            for index, (key, item) in enumerate(value.items()):
                if index >= 4:
                    break
                items.append(dict(key=_sample(key, depth + 1), value=_sample(item, depth + 1)))
            out['items'] = items
        elif type(value) is not set:
            out['items'] = [_sample(item, depth + 1) for item in value[:4]]
        # Sets are not consumed for a fabricated ordering. Generators and other
        # iterables are never traversed by diagnostic inspection.
        out['complete'] = len(value) <= 4 and type(value) is not set
        return out
    return dict(type=kind, complete=False)


class TraceTransformer(ast.NodeTransformer):
    """Instrument only approved ASTs; each original expression evaluates once."""
    def __init__(self, source):
        self.source = source
        self.function = None
        self.sites = {}

    def _site(self, node, kind):
        identity = '%s:%s:%s:%s' % (self.function, node.lineno, node.col_offset, kind)
        self.sites[identity] = dict(function=self.function, line=node.lineno,
                                    column=node.col_offset, kind=kind,
                                    expression=(ast.get_source_segment(self.source, node) or '')[:240])
        if getattr(node, '_default_consumers', None):
            self.sites[identity]['default_consumers'] = list(node._default_consumers)
        return ast.Constant(identity)

    @staticmethod
    def _call(name, args, node, keywords=None):
        return ast.copy_location(ast.Call(func=ast.Name(id=name, ctx=ast.Load()),
                                          args=args, keywords=keywords or []), node)

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node.name
        self.generic_visit(node)
        self.function = previous
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('get', 'sort'):
            site = self._site(node, 'dict_get' if node.func.attr == 'get' else 'inplace_sort')
            if node.func.attr == 'get' and len(node.args) > 1:
                # Static expression ancestry only. Whether the parent actually
                # used this default is a separately recorded execution fact.
                for child in ast.walk(node.args[1]):
                    if isinstance(child, ast.Call):
                        child._default_consumers = getattr(child, '_default_consumers', []) + [site.value]
            self.generic_visit(node)
            # Evaluate the original bound-method expression before its original
            # arguments, including failing attribute lookups, as Python does.
            return self._call('_diag_call', [site, node.func] + node.args, node, node.keywords)
        if isinstance(node.func, ast.Name) and node.func.id == 'sorted':
            site = self._site(node, 'sorted')
            self.generic_visit(node)
            return self._call('_diag_call', [site, node.func] + node.args, node, node.keywords)
        return self.generic_visit(node)

    def visit_Assign(self, node):
        site = self._site(node, 'assignment')
        self.generic_visit(node)
        node.value = self._call('_diag_value', [site, node.value], node.value)
        return node

    def visit_If(self, node):
        site = self._site(node.test, 'condition')
        self.generic_visit(node)
        node.test = self._call('_diag_condition', [site, node.test], node.test)
        return node

    def visit_IfExp(self, node):
        site = self._site(node.test, 'condition')
        self.generic_visit(node)
        node.test = self._call('_diag_condition', [site, node.test], node.test)
        return node

    def visit_Return(self, node):
        site = self._site(node, 'return')
        self.generic_visit(node)
        value = node.value if node.value is not None else ast.Constant(None)
        local_values = self._call('_diag_locals', [], node)
        node.value = self._call('_diag_return', [site, value, local_values], node)
        return node

    def visit_For(self, node):
        site = self._site(node.iter, 'iteration')
        self.generic_visit(node)
        node.iter = self._call('_diag_iter', [site, node.iter], node.iter)
        return node


class ExecutionTrace:
    """Trusted bounded collector. Counters cover recorded sites, samples do not."""
    def __init__(self, sites, operation):
        self.sites = sites
        self.operation = operation
        self.events = {}
        self.omitted_site_calls = 0
        self.sample_bytes = 0
        self.samples_omitted = 0

    def _event(self, site):
        if site not in self.events:
            if len(self.events) >= TRACE_SITES:
                self.omitted_site_calls += 1
                return None
            self.events[site] = dict(site_id=site, **self.sites[site], calls=0, samples=[])
        event = self.events[site]
        event['calls'] += 1
        return event

    def _take(self, event, builder):
        if event is None:
            return
        if len(event['samples']) >= SAMPLES_PER_SITE or self.sample_bytes >= 24 * 1024:
            self.samples_omitted += 1
            return
        value = builder()
        encoded = _wire(value)
        # A large row/locals snapshot does not silently replace other sites.
        if len(encoded.encode()) > 4096:
            value = dict(sample_omitted=True, reason='sample byte budget',
                         serialized_bytes=len(encoded.encode()))
            encoded = _wire(value)
        self.sample_bytes += len(encoded.encode())
        event['samples'].append(value)

    def value(self, site, value):
        event = self._event(site)
        self._take(event, lambda: dict(value=_sample(value)))
        return value

    def condition(self, site, value):
        event = self._event(site)
        if event is not None:
            key = 'true_count' if bool(value) else 'false_count'
            event[key] = event.get(key, 0) + 1
            self._take(event, lambda: dict(value=_sample(value), truth=bool(value)))
        return value

    def returned(self, site, value, local_values):
        event = self._event(site)
        if event is not None and type(value) in (str, list, tuple, dict, set) and len(value) == 0:
            event['empty_count'] = event.get('empty_count', 0) + 1
        def sample():
            names = [name for name in local_values if not name.startswith('_')][:12]
            return dict(value=_sample(value), locals={name: _sample(local_values[name]) for name in names},
                        local_names_count=len(local_values), locals_complete=len(names) == len(local_values))
        self._take(event, sample)
        return value

    def iterate(self, site, iterable):
        # Observe only values consumed by the original loop, never peek ahead
        # or materialize a one-shot/reversed iterator. BudgetedLoops still owns
        # its original iteration accounting around this wrapper.
        for value in iterable:
            event = self._event(site)
            self._take(event, lambda: dict(value=_sample(value)))
            yield value

    def call(self, site, method, /, *args, **kwargs):
        kind = self.sites[site]['kind']
        event = self._event(site)
        receiver = getattr(method, '__self__', None)
        sampling = event is not None and len(event['samples']) < SAMPLES_PER_SITE and self.sample_bytes < 24 * 1024
        before = None
        present = None
        if kind == 'dict_get' and type(receiver) is dict and args:
            # Restrict inspection to hashable primitive keys. The method itself
            # remains responsible for errors from malformed/unhashable keys.
            if type(args[0]) in (str, int, float, bool, type(None)):
                present = args[0] in receiver
                if event is not None:
                    key = 'present_count' if present else 'missing_count'
                    event[key] = event.get(key, 0) + 1
            if sampling:
                before = dict(key=_sample(args[0]), key_present=present,
                              receiver_keys=[_sample(key) for key in list(receiver)[:12]],
                              receiver_key_count=len(receiver),
                              default=_sample(args[1] if len(args) > 1 else None),
                              default_argument_evaluated=len(args) > 1,
                              default_used=(not present) if present is not None else None)
        elif sampling and kind in ('sorted', 'inplace_sort'):
            target = receiver if kind == 'inplace_sort' else args[0] if args else None
            before = dict(input=_sample(target), reverse=_sample(kwargs.get('reverse', False)))
        # Do not catch or replace exceptions raised by the original expression.
        result = method(*args, **kwargs)
        if before is not None:
            before['result'] = _sample(result)
            if kind == 'inplace_sort':
                before['after'] = _sample(receiver)
            self._take(event, lambda: before)
        return result

    def export(self, byte_budget=TRACE_BYTES):
        events = list(self.events.values())
        output = dict(version='execution-observations-v1', operation=self.operation,
                      events=events, coverage=dict(
                          instrumentation=['dict.get', 'sorted', 'list.sort', 'assignment', 'if', 'return', 'for iteration'],
                          uninstrumented=['subscripts', 'comprehension filters', 'and/or operands', 'other methods'],
                          site_limit=TRACE_SITES, samples_per_site=SAMPLES_PER_SITE,
                          recorded_sites=len(events), omitted_site_calls=self.omitted_site_calls,
                          samples_omitted=self.samples_omitted, export_sites_omitted=0,
                          sample_values_are_bounded=True,
                          counters_cover='executions of retained sites only',
                          capability_replay='only final completed worker replay is retained'))
        while len(_wire(output).encode()) > min(TRACE_BYTES, byte_budget) and output['events']:
            output['events'].pop()
            output['coverage']['export_sites_omitted'] += 1
        return output


def _plain_sample(sample):
    """Small actual values, retaining explicit prefix markers for truncation."""
    if 'value' in sample:
        if sample.get('complete'):
            return sample['value']
        return dict(prefix=sample['value'], length=sample.get('length'), complete=False)
    if 'items' in sample and sample.get('type') in ('list', 'tuple'):
        values = [_plain_sample(item) for item in sample['items']]
        if sample.get('complete') and all(item.get('complete') for item in sample['items']):
            return values
        return dict(prefix=values, length=sample.get('length'), complete=False)
    return dict({key: sample[key] for key in ('type', 'length') if key in sample}, complete=False)


def bounded_execution_facts(data, tokenizer, token_budget=2000):
    """Retain whole observed facts for either role, with their original pointers.

    This is shared evidence selection, not a review or a mechanism suggestion.
    Facts from a list remain a prefix so their JSON pointers keep their meaning.
    """
    from evomem.common import digest, dumps
    if not isinstance(data, dict) or not isinstance(data.get('facts'), dict):
        raise ValueError('Behavior diagnostics require a structured facts object')
    base = dict(format='retained_execution_facts_v1', original_hash=digest(data),
                provenance=copy.deepcopy(data.get('provenance', {})), facts={},
                limitations=copy.deepcopy(data.get('limitations', [])),
                omitted_fact_paths=[], clipped=False)
    for key in ('version', 'coverage', 'interpretation_boundary'):
        if key in data:
            base[key] = copy.deepcopy(data[key])
    groups = ['summary', 'missing_dict_keys', 'sorting', 'iteration_prefixes',
              'branch_counts', 'empty_returns', 'adapter_counts']
    groups += sorted(set(data['facts']) - set(groups))
    units = []
    for key in groups:
        if key not in data['facts']:
            continue
        value = data['facts'][key]
        path = '/facts/' + key.replace('~', '~0').replace('/', '~1')
        if isinstance(value, list):
            units.extend((key, index, path + '/' + str(index), item)
                         for index, item in enumerate(value))
            if not value:
                units.append((key, None, path, value))
        else:
            units.append((key, None, path, value))
    base['omitted_fact_paths'] = [unit[2] for unit in units]
    if tokenizer.count(dumps(base)) > token_budget:
        return None
    blocked = set()
    for key, index, path, value in units:
        if key in blocked:
            continue
        candidate = copy.deepcopy(base)
        if index is None:
            candidate['facts'][key] = copy.deepcopy(value)
        else:
            candidate['facts'].setdefault(key, []).append(copy.deepcopy(value))
        candidate['omitted_fact_paths'].remove(path)
        candidate['clipped'] = bool(candidate['omitted_fact_paths'])
        if tokenizer.count(dumps(candidate)) <= token_budget:
            base = candidate
        elif index is not None:
            blocked.add(key)
    base['clipped'] = bool(base['omitted_fact_paths'])
    return base if base['facts'] else None


def iter_execution_receipts(trace, prefix=''):
    """Yield each actual event receipt once with its original JSON pointer.

    Accept one memory invocation or a native memory_hook summary. Nested event
    receipts take precedence over convenience fields on their outer record.
    Event kind describes the lifecycle; model broker operation remains event.
    """
    for key in ('event_trace', 'hook_trace', 'events'):
        if key in trace:
            for index, row in enumerate(trace[key]):
                receipt = row.get('receipt')
                if isinstance(receipt, dict):
                    yield receipt, prefix + '/' + key + '/' + str(index) + '/receipt', row.get('event', {}).get('type', 'event')
            return
    yield trace, prefix, 'event'


def _input_shape(behavior, pointer):
    """Visible input shape from this exact recorded event, without task text."""
    try:
        node = behavior
        for part in pointer.rsplit('/receipt', 1)[0].strip('/').split('/'):
            node = node[int(part)] if isinstance(node, list) else node[part]
        visible = node.get('event', {}).get('visible')
        if not isinstance(visible, dict):
            return None
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return {key: dict(type=type(value).__name__,
                      **({'length': len(value), 'nonempty': bool(value)}
                         if type(value) in (str, list, dict) else {}))
            for key, value in list(visible.items())[:16]}


def _witness_value(sample):
    if not isinstance(sample, dict):
        return None
    result = {key: sample[key] for key in ('type', 'length', 'complete') if key in sample}
    value = sample.get('value')
    if type(value) in (type(None), bool, int, float) and 'value' in sample:
        result['value'] = value
    elif isinstance(value, str):
        result['whitespace_only'] = not value.strip()
        if sample.get('complete') is True and len(value) <= 24:
            result['value'] = value
        else:
            result['recorded_sample_hash'] = _hash(sample)
    return result


def _assignment_witnesses(item, grouped):
    """Recorded assignment samples related by names, not values at later uses."""
    try:
        names = {node.id for node in ast.walk(ast.parse(item['expression'], mode='eval'))
                 if isinstance(node, ast.Name)}
    except (SyntaxError, ValueError):
        return []
    candidates = sorted((other for other in grouped.values()
        if other['kind'] == 'assignment' and other['function'] == item['function']
        and other['operation'] == item['operation']
        and other['first_sample_task_id'] == item['first_sample_task_id']), key=lambda other: other['line'])
    result = []
    for other in candidates:
        try:
            statement = ast.parse(other['expression']).body[0]
        except (SyntaxError, ValueError, IndexError):
            continue
        if not isinstance(statement, ast.Assign):
            continue
        outputs = {node.id for target in statement.targets for node in ast.walk(target)
                   if isinstance(node, ast.Name)}
        inputs = {node.id for node in ast.walk(statement.value) if isinstance(node, ast.Name)}
        if not (outputs & names or inputs & names):
            continue
        names.update(outputs)
        sample = other['first_samples'][0] if other['first_samples'] else {}
        value = _witness_value(sample.get('value'))
        if value is not None:
            result.append(dict(function=other['function'], line=other['line'],
                operation=other['operation'],
                expression=other['expression'], sample_at_assignment=value,
                task_id=other['first_sample_task_id'], behavior_pointer=other['behavior_pointer'],
                boundary='First recorded assignment sample; not a claim about the value at a later use or other invocation'))
        if len(result) == 4:
            break
    return result


def diagnose(program, behavior, logs):
    """Compact facts from recorded replay; never execute code or infer quality.

    The full per-invocation trace stays in the independently saved ``behavior``
    artifact. These deterministic summaries may be quoted by exact JSON pointer;
    the first sample is one observed invocation, not a universal assertion.
    """
    environments = {ep.get('environment') for ep in logs if ep.get('environment') is not None}
    for item in behavior.get('read_trace', []):
        environment = item.get('context', {}).get('environment')
        if environment is not None:
            environments.add(environment)
    if len(environments) > 1:
        raise ValueError('diagnostic evidence cannot mix environments')
    coverage = dict(invocations=0, instrumented_invocations=0, missing_trace_invocations=0,
                    omitted_site_calls=0, export_sites_omitted=0, facts_omitted=0,
                    first_samples_are_not_exhaustive=True, diagnostic_failures=[])
    grouped = {}
    by_kind = {}
    adapter = []
    invocations = []
    for name in ('read_trace', 'write_trace'):
        for index, invocation in enumerate(behavior.get(name, [])):
            prefix = '/' + name + '/' + str(index)
            for receipt, pointer, kind in iter_execution_receipts(invocation, prefix):
                invocations.append((kind, index, invocation, receipt, pointer))
            if name == 'read_trace':
                adapter.append(dict(task_id=invocation.get('context', {}).get('task_id'),
                    raw_count=len(invocation.get('raw_readout', [])),
                    injected_count=len(invocation.get('unpacked_readout', [])),
                    read_tokens=invocation.get('read_tokens')))
    for operation, index, invocation, receipt, pointer in invocations:
        coverage['invocations'] += 1
        trace = receipt.get('execution_trace')
        if not trace:
            coverage['missing_trace_invocations'] += 1
            status = receipt.get('diagnostics')
            if status and status.get('status') == 'unavailable':
                coverage['diagnostic_failures'].append(dict(operation=operation, invocation_index=index,
                                                           receipt=status))
            continue
        coverage['instrumented_invocations'] += 1
        for key in ('omitted_site_calls', 'export_sites_omitted'):
            coverage[key] += trace['coverage'].get(key, 0)
        task_id = invocation.get('task_id', invocation.get('context', {}).get('task_id'))
        for event_index, event in enumerate(trace.get('events', [])):
            kind = event['kind']
            by_kind[kind] = by_kind.get(kind, 0) + event['calls']
            key = (operation, event['site_id'])
            if key not in grouped:
                grouped[key] = dict(operation=operation, function=event['function'],
                    line=event['line'], kind=kind, expression=event['expression'][:100],
                    calls=0, missing_count=0, true_count=0, false_count=0, empty_count=0,
                    first_sample_task_id=task_id, first_samples=event['samples'],
                    behavior_pointer=pointer + '/execution_trace/events/' + str(event_index))
            target = grouped[key]
            for field in ('calls', 'missing_count', 'true_count', 'false_count', 'empty_count'):
                target[field] += event.get(field, 0)
            # The first absent-key sample is more precise than a present
            # key sample from the same executed site; no task scores used.
            if kind == 'dict_get':
                for sample in event['samples']:
                    if sample.get('key_present') is False and 'first_missing' not in target:
                        target['first_missing'] = sample
                        target['first_missing_task_id'] = task_id
                        target['missing_behavior_pointer'] = pointer + '/execution_trace/events/' + str(event_index)
                        target['first_missing_event_visible'] = _input_shape(behavior, pointer)
                        consumers = []
                        for consumer in trace.get('events', []):
                            if consumer['site_id'] not in event.get('default_consumers', []):
                                continue
                            consumers.append(dict(function=consumer['function'], line=consumer['line'],
                                expression=consumer['expression'], calls=consumer['calls'],
                                present_count=consumer.get('present_count', 0), missing_count=consumer.get('missing_count', 0),
                                task_id=task_id,
                                samples=[dict(key=_plain_sample(sample['key']), key_present=sample['key_present'],
                                    result=_plain_sample(sample['result']), default_used=sample.get('default_used'))
                                    for sample in consumer.get('samples', []) if 'key_present' in sample],
                                boundary='Enclosing default-expression call in the same recorded invocation; bounded samples are not a per-loop causal pairing'))
                        target['enclosing_default_calls'] = consumers
    compact_branches = {}
    compact_missing = {}
    for item in grouped.values():
        site = (item['function'], item['line'])
        if item['kind'] == 'condition':
            row = compact_branches.setdefault(site, dict(function=item['function'], line=item['line'],
                                                          true_count=0, false_count=0))
            row['true_count'] += item['true_count']
            row['false_count'] += item['false_count']
        if item['missing_count']:
            row = compact_missing.setdefault(site, dict(function=item['function'], line=item['line'], missing_count=0))
            row['missing_count'] += item['missing_count']
    facts = dict(summary=dict(calls_by_kind=by_kind, retained_executed_sites=len(grouped),
                              branch_counts=list(compact_branches.values()),
                              missing_key_counts=list(compact_missing.values()),
                              compact_tables_aggregate_events=True,
                              sampled_values='first recorded samples; lengths and incomplete prefixes are explicit',
                              detail_order='on_event before helpers; protocol event order; ascending source line',
                              counter_scope='all retained sites in recorded offline invocations'),
                 missing_dict_keys=[], sorting=[], iteration_prefixes=[], branch_counts=[],
                 empty_returns=[], adapter_counts=adapter[:4])
    limits = dict(missing_dict_keys=4, sorting=3, iteration_prefixes=3, branch_counts=16, empty_returns=3)
    def add(kind, value):
        if len(facts[kind]) < limits[kind]:
            facts[kind].append(value)
        else:
            coverage['facts_omitted'] += 1
    event_order = {name: index for index, name in enumerate(EVENTS)}
    ordered = sorted(grouped.values(), key=lambda item: (item['function'] != 'on_event',
        event_order.get(item['operation'], len(EVENTS)), item['line']))
    for item in ordered:
        common = {key: item[key] for key in ('operation', 'function', 'line', 'expression', 'calls')}
        if item['missing_count']:
            example = item.get('first_missing')
            value = dict(common, missing_count=item['missing_count'], first_missing=None)
            if example is not None:
                value['first_missing'] = dict(task_id=item['first_missing_task_id'],
                    key=_plain_sample(example['key']), default=_plain_sample(example['default']),
                    result=_plain_sample(example['result']),
                    receiver_keys=[_plain_sample(key) for key in example['receiver_keys']],
                    receiver_key_count=example['receiver_key_count'],
                    default_argument_evaluated=example.get('default_argument_evaluated'),
                    default_used=example.get('default_used'),
                    behavior_pointer=item['missing_behavior_pointer'])
                value['event_visible_shape'] = item.get('first_missing_event_visible')
                if item.get('enclosing_default_calls'):
                    value['enclosing_default_calls'] = item['enclosing_default_calls']
            value['assignment_witnesses'] = _assignment_witnesses(item, grouped)
            add('missing_dict_keys', value)
        if item['kind'] in ('sorted', 'inplace_sort'):
            sample = item['first_samples'][0] if item['first_samples'] else None
            values = {key: _plain_sample(value) for key, value in sample.items()} if sample and not sample.get('sample_omitted') else None
            add('sorting', dict(common, kind=item['kind'], first_sample_task_id=item['first_sample_task_id'],
                                first_sample=values, behavior_pointer=item['behavior_pointer']))
        if item['kind'] == 'iteration':
            add('iteration_prefixes', dict(common, first_sample_task_id=item['first_sample_task_id'],
                values=[_plain_sample(sample['value']) for sample in item['first_samples'] if 'value' in sample],
                behavior_pointer=item['behavior_pointer']))
        if item['kind'] == 'condition':
            add('branch_counts', dict(common, true_count=item['true_count'], false_count=item['false_count'],
                                     assignment_witnesses=_assignment_witnesses(item, grouped)))
        if item['empty_count']:
            add('empty_returns', dict(common, empty_count=item['empty_count']))
    coverage['facts_omitted'] += max(0, len(adapter) - 4)
    return dict(version='behavior-diagnostics-v2',
                provenance=dict(program_hash=program.hash, environment=next(iter(environments), None),
                                phase='offline_replay', snapshot_hash=behavior.get('memory_hash') or _hash(behavior.get('stored_state', {}))),
                facts=facts, coverage=coverage,
                interpretation_boundary='Observed execution only; missing keys, empty values, ordering and task reward do not by themselves establish a defect or mechanism quality. A default argument is evaluated even when its enclosing dict.get returns a present key; missing fallback fields do not prove an empty query.')
