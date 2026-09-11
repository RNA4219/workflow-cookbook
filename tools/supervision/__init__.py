"""Cross-process supervision helpers for shared workspaces.

Public objects live in :mod:`tools.supervision.workspace_coordinator`.  The
package intentionally avoids eager imports so ``python -m`` can execute the
CLI without loading the command module twice.
"""
