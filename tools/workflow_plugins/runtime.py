# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .plugin_config import load_workflow_plugin_specs
from .errors import WorkflowPluginCapabilityError, WorkflowPluginTimeoutError, WorkflowPluginExecutionError
from .interfaces import CAPABILITY_METHOD_NAMES
from .plugin_loader import instantiate_workflow_plugins


@dataclass
class PluginPolicy:
    """Policy for plugin execution."""
    timeout_seconds: float = 30.0
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    continue_on_error: bool = False
    trace_enabled: bool = False


@dataclass
class PluginTrace:
    """Trace record for a plugin invocation."""
    plugin_name: str
    capability: str
    method_name: str
    start_time: float
    end_time: float | None = None
    success: bool = True
    error: str | None = None
    result_summary: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class InvocationResult:
    """Result of a plugin invocation with tracing info."""
    result: Any
    trace: PluginTrace
    error: Exception | None = None


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
    ) -> "WorkflowPluginRuntime":
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
        trace = PluginTrace(
            plugin_name=plugin_name,
            capability=capability,
            method_name=method_name,
            start_time=time.time(),
        )

        method = getattr(plugin, method_name)
        last_error: Exception | None = None

        for attempt in range(policy.retry_count + 1):
            try:
                if policy.timeout_seconds > 0:
                    # Simple timeout using time check
                    start = time.time()
                    result = method(**kwargs)
                    elapsed = time.time() - start
                    if elapsed > policy.timeout_seconds:
                        raise WorkflowPluginTimeoutError(
                            f"Plugin {plugin_name}.{method_name} exceeded "
                            f"timeout of {policy.timeout_seconds}s"
                        )
                else:
                    result = method(**kwargs)

                trace.end_time = time.time()
                trace.success = True
                if policy.trace_enabled:
                    self._traces.append(trace)
                return InvocationResult(result=result, trace=trace)

            except WorkflowPluginTimeoutError:
                raise
            except Exception as e:
                last_error = e
                trace.error = str(e)
                trace.success = False

                if attempt < policy.retry_count:
                    time.sleep(policy.retry_delay_seconds)
                else:
                    trace.end_time = time.time()
                    if policy.trace_enabled:
                        self._traces.append(trace)
                    if not policy.continue_on_error:
                        raise WorkflowPluginExecutionError(
                            f"Plugin {plugin_name}.{method_name} failed: {e}"
                        ) from e

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