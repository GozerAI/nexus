"""Nexus Intelligence GUI - Desktop Application."""

import os

# Fix potential Qt rendering issues that can cause segmentation faults
# These settings must be set BEFORE importing any Qt modules
if 'QT_QUICK_BACKEND' not in os.environ:
    os.environ['QT_QUICK_BACKEND'] = 'software'
if 'QSG_RENDER_LOOP' not in os.environ:
    os.environ['QSG_RENDER_LOOP'] = 'basic'

from nexus.gui.app import NexusApp, run

__all__ = ["NexusApp", "run"]
