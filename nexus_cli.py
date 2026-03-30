#!/usr/bin/env python
"""Nexus CLI entry point."""
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

from nexus.cli.main import cli

if __name__ == '__main__':
    cli()
