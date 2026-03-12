"""
Nexus Intelligence Platform Launcher (No Console Window)
Starts Redis if available, then launches the GUI.
"""

import sys
import os
import subprocess
import socket
from pathlib import Path

# Get the project root
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Change to project directory
os.chdir(project_root)

# Check if Redis is running
def is_redis_running():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 6379))
        sock.close()
        return result == 0
    except:
        return False

# Start Redis if installed and not running
redis_server = project_root / "tools" / "redis" / "redis-server.exe"
redis_config = project_root / "tools" / "redis" / "redis-nexus.conf"

if redis_server.exists() and not is_redis_running():
    subprocess.Popen(
        [str(redis_server), str(redis_config)],
        cwd=str(redis_server.parent),
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # Give Redis a moment to start
    import time
    time.sleep(2)

# Find Python executable (prefer venv)
venv_python = project_root / ".venv" / "Scripts" / "python.exe"
if not venv_python.exists():
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
if not venv_python.exists():
    venv_python = sys.executable

# Launch the GUI with src in PYTHONPATH
env = os.environ.copy()
src_dir = project_root / "src"
existing_path = env.get("PYTHONPATH", "")
env["PYTHONPATH"] = str(src_dir) + (os.pathsep + existing_path if existing_path else "")

subprocess.Popen(
    [str(venv_python), "-m", "nexus.gui"],
    cwd=str(project_root),
    env=env,
    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
)
