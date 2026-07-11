# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .errors import WorkflowPluginCapabilityError, WorkflowPluginExecutionError, WorkflowPluginTimeoutError
from .interfaces import CAPABILITY_METHOD_NAMES
from .plugin_config import load_workflow_plugin_specs
from .plugin_loader import instantiate_workflow_plugins
from .runtime_evidence import (
    build_trace_evidence_payload,
    trace_payload,
    write_trace_evidence_jsonl,
    write_trace_payload,
)
from .runtime_types import InvocationResult, PluginPolicy, PluginTrace


class WorkflowPluginRuntime:
    def __init__(
        self,
        plugins: Sequence[Any],
        *,
        default_policy: PluginPolicy | None = None,
        capability_policies: dict[str, PluginPolicy] | None = None,
    ) -> None:
        self._plugins = list(plugins)
        self._default_policy = default_policy or PluginPolicy()
        self._capability_policies = capability_policies or {}
        self._traces: list[PluginTrace] = []

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        default_policy: PluginPolicy | None = None,
        capability_policies: dict[str, PluginPolicy] | None = None,
    ) -> WorkflowPluginRuntime:
        target = Path(config_path).expanduser().resolve()
        specs = load_workflow_plugin_specs(target)
        plugins = instantiate_workflow_plugins(specs, base_path=target.parent)
        return cls(
            plugins,
            default_policy=default_policy,
            capability_policies=capability_policies,
        )

    @property
    def plugins(self) -> list[Any]:
        return list(self._plugins)

    @property
    def traces(self) -> list[PluginTrace]:
        return list(self._traces)

    def trace_payload(self) -> list[dict[str, Any]]:
        return trace_payload(self._traces)

    def write_traces_json(self, path: str | Path) -> None:
        write_trace_payload(path, self._traces)

    def trace_evidence_payload(
        self,
        *,
        task_seed_id: str,
        base_commit: str,
        head_commit: str,
        actor: str,
        evidence_id_prefix: str = "EV-WORKFLOW-PLUGIN",
    ) -> list[dict[str, Any]]:
        return build_trace_evidence_payload(
            self._traces,
            task_seed_id=task_seed_id,
            base_commit=base_commit,
            head_commit=head_commit,
            actor=actor,
            evidence_id_prefix=evidence_id_prefix,
        )

    def write_trace_evidence_jsonl(
        self,
        path: str | Path,
        *,
        task_seed_id: str,
        base_commit: str,
        head_commit: str,
        actor: str,
        evidence_id_prefix: str = "EV-WORKFLOW-PLUGIN",
    ) -> None:
        write_trace_evidence_jsonl(
            path,
            self._traces,
            task_seed_id=task_seed_id,
            base_commit=base_commit,
            head_commit=head_commit,
            actor=actor,
            evidence_id_prefix=evidence_id_prefix,
        )

    def clear_traces(self) -> None:
        self._traces.clear()

    def get_policy(self, capability: str) -> PluginPolicy:
        return self._capability_policies.get(capability, self._default_policy)

    def iter_capability(self, capability: str) -> Iterable[Any]:
        for plugin in self._plugins:
            capabilities = getattr(plugin, "capabilities", ())
            if capability in capabilities:
                yield plugin

    def first_capability(self, capability: str) -> Any:
        for plugin in self.iter_capability(capability):
            return plugin
        raise WorkflowPluginCapabilityError(f"No workflow plugin provides {capability}.")

    def _call_method(
        self,
        method: Callable[..., Any],
        kwargs: dict[str, Any],
        *,
        plugin_name: str,
        method_name: str,
        policy: PluginPolicy,
    ) -> Any:
        if policy.timeout_seconds <= 0 or policy.isolation_mode == "inline":
            return method(**kwargs)
        if policy.isolation_mode != "thread":
            raise WorkflowPluginExecutionError(
                f"Unsupported workflow plugin isolation mode: {policy.isolation_mode}"
            )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(method, **kwargs)
        try:
            result = future.result(timeout=policy.timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise WorkflowPluginTimeoutError(
                f"Plugin {plugin_name}.{method_name} exceeded timeout of {policy.timeout_seconds}s"
            ) from exc
        else:
            executor.shutdown(wait=True, cancel_futures=False)
            return result

    def _append_trace(self, trace: PluginTrace, policy: PluginPolicy) -> None:
        if policy.trace_enabled:
            self._traces.append(trace)

    @staticmethod
    def _summarize_result(result: Any) -> str:
        if result is None:
            return "None"
        if isinstance(result, dict):
            return f"dict(keys={sorted(str(key) for key in result.keys())})"
        if isinstance(result, (list, tuple, set)):
            return f"{type(result).__name__}(len={len(result)})"
        return type(result).__name__

    def _invoke_with_policy(
        self,
        plugin: Any,
        method_name: str,
        capability: str,
        kwargs: dict[str, Any],
        policy: PluginPolicy,
    ) -> InvocationResult:
        """Invoke a plugin method with timeout and retry policy."""
        plugin_name = getattr(plugin, "__class__", type(plugin)).__name__
        method = getattr(plugin, method_name)
        last_error: Exception | None = None

        for attempt in range(policy.retry_count + 1):
            trace = PluginTrace(
                plugin_name=plugin_name,
                capability=capability,
                method_name=method_name,
                start_time=time.time(),
                attempt=attempt + 1,
                timeout_seconds=policy.timeout_seconds if policy.timeout_seconds > 0 else None,
                isolation_mode=policy.isolation_mode,
            )
            try:
                result = self._call_method(
                    method,
                    kwargs,
                    plugin_name=plugin_name,
                    method_name=method_name,
                    policy=policy,
                )

                trace.end_time = time.time()
                trace.success = True
                trace.result_summary = self._summarize_result(result)
                self._append_trace(trace, policy)
                return InvocationResult(result=result, trace=trace)

            except Exception as e:
                last_error = e
                trace.error = str(e)
                trace.success = False
                trace.timed_out = isinstance(e, WorkflowPluginTimeoutError)

                if attempt < policy.retry_count:
                    trace.end_time = time.time()
                    self._append_trace(trace, policy)
                    time.sleep(policy.retry_delay_seconds)
                else:
                    trace.end_time = time.time()
                    self._append_trace(trace, policy)
                    if not policy.continue_on_error:
                        if isinstance(e, WorkflowPluginTimeoutError):
                            raise e
                        raise WorkflowPluginExecutionError(f"Plugin {plugin_name}.{method_name} failed: {e}") from e

        # Should not reach here, but satisfy type checker
        trace.end_time = time.time()
        return InvocationResult(result=None, trace=trace, error=last_error)

    def invoke_first(
        self,
        capability: str,
        *,
        coercer: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        plugin = self.first_capability(capability)
        method_name = CAPABILITY_METHOD_NAMES[capability]
        policy = self.get_policy(capability)

        invocation = self._invoke_with_policy(plugin, method_name, capability, kwargs, policy)

        if invocation.error and not policy.continue_on_error:
            raise invocation.error

        if coercer is None:
            return invocation.result
        return coercer(invocation.result)

    def invoke_all(
        self,
        capability: str,
        *,
        coercer: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        results: list[Any] = []
        policy = self.get_policy(capability)

        for plugin in self.iter_capability(capability):
            method_name = CAPABILITY_METHOD_NAMES[capability]
            invocation = self._invoke_with_policy(plugin, method_name, capability, kwargs, policy)

            if invocation.error and not policy.continue_on_error:
                raise invocation.error

            result = invocation.result
            if coercer is not None:
                result = coercer(result)
            results.append(result)

        return results


__all__ = [
    "WorkflowPluginRuntime",
    "PluginPolicy",
    "PluginTrace",
    "InvocationResult",
]
