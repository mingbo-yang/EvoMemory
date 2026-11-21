"""Task-local memory events around the unchanged official ALMA loop.

The official method's code object is reused with a private globals dictionary
and an episode-local environment proxy. No ALMA module/class/global Agent is
patched. Only the workflow instance has a temporary, context-aware dispatcher.
"""
from __future__ import annotations

from contextvars import ContextVar
import copy
import inspect
import threading
import types

from evomem.alma_backend import experience, public_context
from .memory import digest
from .policy import Rejected, json_copy


active_hook = ContextVar('phase1_active_memory_hook', default=None)
_active_adapter = ContextVar('phase1_active_memory_hook_adapter', default=None)
_installation_lock = threading.RLock()
_INSTALLATION = '_evomem_event_dispatcher'


class HookExecutionError(Rejected):
    """A memory-hook failure caught outside the original action-loop handler."""


class _HookAbort(BaseException):
    # ALMA catches Exception inside its action loop and adds error instructions.
    # This private signal crosses that handler, then becomes a normal recorded
    # episode failure at the run_single_task boundary.
    def __init__(self, cause):
        self.cause = cause


def _visible(value):
    result = public_context(value)
    for key in ('scores', 'dones'):
        if key in value:
            result[key] = value[key]
    return json_copy(result)


class _EnvironmentView:
    def __init__(self, owner, environment):
        self.owner = owner
        self.original = environment

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def get_prompt(self, *args, **kwargs):
        prompts = await self.original.get_prompt(*args, **kwargs)
        if self.owner.initial_prompt is None:
            self.owner.initial_prompt = copy.deepcopy(prompts)
            self.owner.initial_prompt_args = copy.deepcopy(args)
            self.owner.initial_prompt_kwargs = copy.deepcopy(kwargs)
            self.owner.visible = _visible(kwargs)
            self.owner.rendered_memory = json_copy(kwargs.get('memory_retrived', {}))
            self.owner.rendered_system = prompts[0]['content']
        return prompts

    async def run_step(self, action, game_files, **kwargs):
        # An actual environment exception remains an original ALMA exception;
        # no observation is fabricated when the native call did not return.
        try:
            new_env, recorder = await self.original.run_step(action, game_files, **kwargs)
        except Exception as exc:
            # ALMA supplies its own feedback to the Actor. Only publish the
            # submitted action and a public category to the subsequent Hook.
            # An arbitrary native exception does not prove an invalid action.
            explicit_rejection = (type(exc) is RuntimeError
                                  and exc.args == (f'Action: {action} is not valid.',))
            self.owner.previous_action_failure = dict(
                attempt=self.owner.attempts - 1, action=action,
                reason='invalid_action' if explicit_rejection else 'step_failed')
            raise
        self.owner.completed_steps += 1
        self.owner.visible = _visible(new_env)
        try:
            self.owner.session.emit('after_observation', dict(
                step=self.owner.completed_steps, attempt=self.owner.attempts - 1,
                visible=self.owner.visible, action=action))
        except Exception as exc:
            raise _HookAbort(exc) from exc
        return new_env, recorder


class _WorkflowView:
    def __init__(self, original, environment):
        self.original = original
        self.env = environment

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Installation:
    def __init__(self, workflow):
        self.workflow = workflow
        self.original = workflow.run_single_task
        if not inspect.ismethod(self.original) or not inspect.iscoroutinefunction(self.original):
            raise Rejected('Hook adapter requires the official async bound workflow method')
        self.had_local_method = 'run_single_task' in vars(workflow)
        self.local_method = vars(workflow).get('run_single_task')
        self.owners = 0

        async def dispatch(*args, **kwargs):
            adapter = _active_adapter.get()
            if adapter is None or adapter.workflow is not workflow:
                return await self.original(*args, **kwargs)
            if active_hook.get() is not adapter.session:
                raise HookExecutionError('Hook episode context does not match its adapter')
            return await adapter._run(self.original, *args, **kwargs)

        self.dispatch = dispatch


class HookAdapter:
    """Use ``with HookAdapter(...): await workflow.run_single_task(...)``.

    The native bridge emits task_start in general_retrieve and records the
    write intent only in general_update. This adapter emits before_action and
    after_observation, then stages task_end after official reward and
    general_update succeed. The native finalizer validates and publishes later.
    session.abort records unsuccessful termination without publishing state.

    ``step`` counts completed native transitions; ``attempt`` is the zero-based
    original Agent.ask call. Native failures/retries therefore remain distinct.
    ``injections`` records the exact replaced slot and message hashes per ask.
    """
    def __init__(self, workflow, session, task, phase, seed):
        self.workflow = workflow
        self.session = session
        self.task = json_copy(task)
        self.phase = phase
        self.seed = seed
        self.attempts = 0
        self.completed_steps = 0
        self.visible = {}
        self.previous_action_failure = None
        self.initial_prompt = None
        self.initial_prompt_args = None
        self.initial_prompt_kwargs = None
        self.rendered_memory = None
        self.rendered_system = None
        self.injections = []
        self.used = False
        self.entered = False
        self.environment = _EnvironmentView(self, workflow.env)

    def __enter__(self):
        if self.entered:
            raise Rejected('HookAdapter cannot be entered twice')
        with _installation_lock:
            installed = getattr(self.workflow, _INSTALLATION, None)
            if installed is None:
                installed = _Installation(self.workflow)
                setattr(self.workflow, _INSTALLATION, installed)
                self.workflow.run_single_task = installed.dispatch
            elif not isinstance(installed, _Installation):
                raise Rejected('workflow Hook dispatcher attribute is already occupied')
            installed.owners += 1
            self.installation = installed
        self.hook_token = active_hook.set(self.session)
        self.adapter_token = _active_adapter.set(self)
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _active_adapter.reset(self.adapter_token)
        active_hook.reset(self.hook_token)
        with _installation_lock:
            self.installation.owners -= 1
            if self.installation.owners == 0:
                if self.installation.had_local_method:
                    self.workflow.run_single_task = self.installation.local_method
                else:
                    delattr(self.workflow, 'run_single_task')
                delattr(self.workflow, _INSTALLATION)
        self.entered = False
        return False

    async def _system_prompt(self):
        if self.initial_prompt is None:
            raise Rejected('official initial prompt was not observed before Actor.ask')
        memory = json_copy(self.session.current_memory)
        if memory != self.rendered_memory:
            kwargs = copy.deepcopy(self.initial_prompt_kwargs)
            kwargs['memory_retrived'] = memory
            prompts = await self.environment.original.get_prompt(*self.initial_prompt_args, **kwargs)
            self.rendered_system = prompts[0]['content']
            self.rendered_memory = memory
        return self.rendered_system

    def _agent_factory(self, original_agent):
        owner = self

        def factory(*args, **kwargs):
            agent = original_agent(*args, **kwargs)
            original_ask = agent.ask
            # Official loop concatenates the environment system prompt with
            # this constructor suffix. Keep the exact same concatenation.
            suffix = agent.messages[0]['content']

            async def ask(instance, user_input, *ask_args, **ask_kwargs):
                attempt = owner.attempts
                owner.attempts += 1
                try:
                    payload = dict(step=owner.completed_steps, attempt=attempt, visible=owner.visible)
                    if owner.previous_action_failure is not None:
                        payload['previous_action_failure'] = owner.previous_action_failure
                        owner.previous_action_failure = None
                    owner.session.emit('before_action', payload)
                    rendered = await owner._system_prompt()
                    instance.messages[0]['content'] = rendered + suffix
                    owner.injections.append(dict(
                        task_id=owner.task['id'], attempt=attempt, step=owner.completed_steps,
                        memory=json_copy(owner.session.current_memory),
                        system_message=instance.messages[0]['content'],
                        system_hash=digest(instance.messages[0]['content']),
                        original_user_prompt_hash=digest(user_input),
                        event_sequence=len(owner.session.trace) - 1))
                except Exception as exc:
                    raise _HookAbort(exc) from exc
                # Exact original input, history flag, reasoning setting and
                # action-generation method. Never postprocess its answer.
                return await original_ask(user_input, *ask_args, **ask_kwargs)

            agent.ask = types.MethodType(ask, agent)
            return agent

        return factory

    async def _run(self, original, *args, **kwargs):
        if self.used:
            raise Rejected('one HookAdapter is limited to one native episode')
        self.used = True
        function = original.__func__
        namespace = dict(function.__globals__)
        if 'Agent' not in namespace:
            raise Rejected('official workflow no longer declares its Agent binding')
        namespace['Agent'] = self._agent_factory(namespace['Agent'])
        invoke = types.FunctionType(function.__code__, namespace, function.__name__,
                                    function.__defaults__, function.__closure__)
        invoke.__kwdefaults__ = function.__kwdefaults__
        # Preserve official bytecode, control flow and local variables. Only
        # object dependencies are task-local; no global monkeypatch is needed.
        bound = types.MethodType(invoke, _WorkflowView(self.workflow, self.environment))
        try:
            recorder = await bound(*args, **kwargs)
            ep = experience(recorder, self.task, self.phase, self.seed)
            ep['environment'] = self.session.context['environment']
            self.session.finish(ep)
            return recorder
        except _HookAbort as exc:
            self.session.abort(str(exc.cause), dict(step=self.completed_steps, attempts=self.attempts))
            raise HookExecutionError('Memory hook failed: ' + str(exc.cause)) from exc.cause
        except BaseException as exc:
            self.session.abort(str(exc), dict(step=self.completed_steps, attempts=self.attempts))
            raise
