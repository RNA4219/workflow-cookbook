# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations


class WorkflowPluginError(ValueError):
    """Base error for workflow plugin host failures."""


class WorkflowPluginLoadError(WorkflowPluginError):
    """Raised when a plugin factory or instance cannot be loaded."""


class WorkflowPluginCapabilityError(WorkflowPluginError):
    """Raised when a plugin does not provide a required capability."""
