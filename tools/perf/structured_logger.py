from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Mapping, Protocol, Sequence

JsonMapping = Mapping[str, Any]


def _utc_timestamp(value: datetime | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    elif value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


@dataclass(slots=True)
class InferenceLogRecord:
    logger: str
    event: str
    level: str
    timestamp: str
    inference_id: str | None = None
    model: str | None = None
    prompt: JsonMapping | None = None
    response: JsonMapping | None = None
    metrics: JsonMapping | None = None
    tags: Sequence[str] | None = None
    extra: JsonMapping | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "logger": self.logger,
            "event": self.event,
            "level": self.level,
            "timestamp": self.timestamp,
            "inference_id": self.inference_id,
            "model": self.model,
            "prompt": _to_jsonable(self.prompt) if self.prompt is not None else None,
            "response": _to_jsonable(self.response) if self.response is not None else None,
            "metrics": _to_jsonable(self.metrics) if self.metrics is not None else {},
            "tags": list(self.tags) if self.tags is not None else [],
            "extra": _to_jsonable(self.extra) if self.extra is not None else {},
        }


class InferenceEvidenceSink(Protocol):
    def emit_inference_evidence(self, record: InferenceLogRecord) -> None:
        ...


class InferenceLogPlugin(Protocol):
    def handle_inference(self, record: InferenceLogRecord) -> None:
        ...


@dataclass(slots=True)
class _LegacyEvidenceSinkAdapter:
    sink: InferenceEvidenceSink

    def handle_inference(self, record: InferenceLogRecord) -> None:
        self.sink.emit_inference_evidence(record)


class StructuredLogger:
    __slots__ = ("_name", "_path", "_stream", "_plugins")

    def __init__(
        self,
        *,
        name: str,
        path: str | Path | None = None,
        stream: IO[str] | None = None,
        evidence_sink: InferenceEvidenceSink | None = None,
        plugins: Sequence[InferenceLogPlugin] | None = None,
    ) -> None:
        if path is None and stream is None:
            raise ValueError("Either path or stream must be provided")
        self._name = name
        self._path = Path(path).expanduser() if path is not None else None
        self._stream = stream
        resolved_plugins: list[InferenceLogPlugin] = list(plugins or ())
        if evidence_sink is not None:
            resolved_plugins.append(_LegacyEvidenceSinkAdapter(evidence_sink))
        self._plugins = tuple(resolved_plugins)

    @classmethod
    def from_plugin_specs(
        cls,
        *,
        name: str,
        plugin_specs: Sequence[object],
        path: str | Path | None = None,
        stream: IO[str] | None = None,
        evidence_sink: InferenceEvidenceSink | None = None,
    ) -> StructuredLogger:
        from tools.protocols.plugin_loader import instantiate_inference_plugins

        plugins = instantiate_inference_plugins(plugin_specs)
        return cls(
            name=name,
            path=path,
            stream=stream,
            evidence_sink=evidence_sink,
            plugins=plugins,
        )

    @classmethod
    def from_plugin_config(
        cls,
        *,
        name: str,
        plugin_config: object,
        path: str | Path | None = None,
        stream: IO[str] | None = None,
        evidence_sink: InferenceEvidenceSink | None = None,
    ) -> StructuredLogger:
        from tools.protocols.plugin_config import load_inference_plugin_specs

        plugin_specs = load_inference_plugin_specs(plugin_config)
        return cls.from_plugin_specs(
            name=name,
            plugin_specs=plugin_specs,
            path=path,
            stream=stream,
            evidence_sink=evidence_sink,
        )

    @property
    def name(self) -> str:
        return self._name

    def inference(
        self,
        *,
        inference_id: str | None = None,
        model: str | None = None,
        prompt: JsonMapping | None = None,
        response: JsonMapping | None = None,
        metrics: JsonMapping | None = None,
        tags: Sequence[str] | None = None,
        extra: JsonMapping | None = None,
        level: str = "INFO",
        timestamp: datetime | None = None,
    ) -> None:
        record = InferenceLogRecord(
            logger=self._name,
            event="inference",
            level=level,
            timestamp=_utc_timestamp(timestamp),
            inference_id=inference_id,
            model=model,
            prompt=prompt,
            response=response,
            metrics=metrics,
            tags=tuple(tags) if tags is not None else None,
            extra=extra,
        )
        self._write(record)

    def _write(self, record: InferenceLogRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        if self._stream is not None:
            self._stream.write(line)
            self._stream.flush()
        else:
            assert self._path is not None
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        for plugin in self._plugins:
            plugin.handle_inference(record)


__all__ = ["InferenceEvidenceSink", "InferenceLogPlugin", "InferenceLogRecord", "StructuredLogger"]
