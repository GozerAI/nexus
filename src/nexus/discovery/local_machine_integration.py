"""
Local Machine Integration - Access local files, system info, and execute commands.

Enables Nexus to:
1. Read and search local files
2. Gather system information (CPU, memory, disk, processes)
3. Execute shell commands (with safety controls)
4. Monitor local resources
5. Access local databases and configurations
"""

import asyncio
import glob
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Security: Dangerous command patterns that could harm the system
DANGEROUS_PATTERNS = [
    r'\brm\s+(-[rf]+\s+)*[/\\]',  # rm -rf / or similar
    r'\bformat\b',  # format drives
    r'\bdel\s+/[fsq]',  # del /f /s /q
    r'\brd\s+/[sq]',  # rd /s /q
    r':\(\)\{.*\};:',  # Fork bomb
    r'\bmkfs\b',  # Make filesystem
    r'\bdd\s+if=',  # dd command
    r'\b>\s*/dev/sd[a-z]',  # Write to raw disk
    r'\bchmod\s+777\s+/',  # Dangerous permission change
    r'\bchown\s+.*\s+/',  # Dangerous ownership change at root
    r';\s*rm\s+',  # Command chaining with rm
    r'\|\s*rm\s+',  # Pipe to rm
    r'`.*rm.*`',  # Backtick command substitution with rm
    r'\$\(.*rm.*\)',  # Command substitution with rm
    r'\bsudo\s+rm\b',  # Sudo rm
    r'\bsudo\s+dd\b',  # Sudo dd
    r'\bwget\s+.*\|\s*(ba)?sh',  # Pipe wget to shell
    r'\bcurl\s+.*\|\s*(ba)?sh',  # Pipe curl to shell
]


class LocalMachineIntegration:
    """
    Local machine integration for accessing files, system info, and running commands.

    Capabilities:
    - File operations: read, search, list directories
    - System info: CPU, memory, disk, network, processes
    - Command execution: run shell commands with safety controls
    - Environment access: env vars, Python packages, paths
    """

    def __init__(
        self,
        allowed_paths: Optional[List[str]] = None,
        blocked_commands: Optional[List[str]] = None,
        enable_command_execution: bool = True,
    ):
        """
        Initialize local machine integration.

        Args:
            allowed_paths: List of allowed base paths for file access (None = all)
            blocked_commands: Commands that are blocked from execution
            enable_command_execution: Whether to allow command execution
        """
        self.allowed_paths = allowed_paths
        self.blocked_commands = blocked_commands or [
            "rm -rf /",
            "format",
            "del /f /s /q",
            "rd /s /q",
            ":(){:|:&};:",  # Fork bomb
        ]
        self.enable_command_execution = enable_command_execution

        logger.info("LocalMachineIntegration initialized")

    # ==================== System Information ====================

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            },
            "cpu": self._get_cpu_info(),
            "memory": self._get_memory_info(),
            "disk": self._get_disk_info(),
            "environment": self._get_environment_summary(),
        }

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        info = {
            "physical_cores": os.cpu_count(),
            "logical_cores": os.cpu_count(),
        }

        try:
            import psutil
            info["physical_cores"] = psutil.cpu_count(logical=False)
            info["logical_cores"] = psutil.cpu_count(logical=True)
            info["percent"] = psutil.cpu_percent(interval=0.1)
            info["freq"] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        except ImportError:
            pass

        return info

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024 ** 3), 2),
                "available_gb": round(mem.available / (1024 ** 3), 2),
                "used_gb": round(mem.used / (1024 ** 3), 2),
                "percent": mem.percent,
            }
        except ImportError:
            return {"error": "psutil not installed"}

    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk information."""
        try:
            import psutil
            partitions = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    partitions.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": round(usage.total / (1024 ** 3), 2),
                        "used_gb": round(usage.used / (1024 ** 3), 2),
                        "free_gb": round(usage.free / (1024 ** 3), 2),
                        "percent": usage.percent,
                    })
                except PermissionError:
                    continue
            return {"partitions": partitions}
        except ImportError:
            # Fallback without psutil
            total, used, free = shutil.disk_usage("/")
            return {
                "partitions": [{
                    "mountpoint": "/",
                    "total_gb": round(total / (1024 ** 3), 2),
                    "used_gb": round(used / (1024 ** 3), 2),
                    "free_gb": round(free / (1024 ** 3), 2),
                }]
            }

    def _get_environment_summary(self) -> Dict[str, Any]:
        """Get environment summary."""
        return {
            "python_executable": sys.executable,
            "python_path": sys.path[:5],  # First 5 entries
            "cwd": os.getcwd(),
            "home": str(Path.home()),
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
            "shell": os.environ.get("SHELL") or os.environ.get("COMSPEC"),
        }

    def get_running_processes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of running processes."""
        try:
            import psutil
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    processes.append({
                        "pid": pinfo['pid'],
                        "name": pinfo['name'],
                        "cpu_percent": pinfo['cpu_percent'],
                        "memory_percent": round(pinfo['memory_percent'], 2) if pinfo['memory_percent'] else 0,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            return processes[:limit]
        except ImportError:
            return [{"error": "psutil not installed"}]

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        try:
            import psutil
            interfaces = {}
            for name, addrs in psutil.net_if_addrs().items():
                interfaces[name] = []
                for addr in addrs:
                    interfaces[name].append({
                        "family": str(addr.family),
                        "address": addr.address,
                        "netmask": addr.netmask,
                    })
            return {"interfaces": interfaces}
        except ImportError:
            return {"error": "psutil not installed"}

    # ==================== File Operations ====================

    def _is_path_allowed(self, path: str) -> bool:
        """
        Check if path is allowed, with protection against path traversal attacks.

        Uses canonical path resolution to prevent symlink and relative path attacks.
        """
        if self.allowed_paths is None:
            return True

        try:
            # Resolve to canonical path (handles symlinks, .., etc.)
            # Use Path.resolve() which is safer than os.path.abspath()
            canonical_path = Path(path).resolve()

            for allowed in self.allowed_paths:
                allowed_canonical = Path(allowed).resolve()
                # Use is_relative_to() for safe path containment check
                try:
                    canonical_path.relative_to(allowed_canonical)
                    return True
                except ValueError:
                    # Not relative to this allowed path, try next
                    continue

            return False
        except (OSError, ValueError) as e:
            logger.warning(f"Path validation error for '{path}': {e}")
            return False

    def read_file(
        self,
        path: str,
        max_size_mb: float = 10,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """
        Read a file's contents.

        Args:
            path: File path
            max_size_mb: Maximum file size in MB
            encoding: File encoding

        Returns:
            Dict with content or error
        """
        try:
            abs_path = os.path.abspath(path)

            if not self._is_path_allowed(abs_path):
                return {"error": f"Path not allowed: {path}"}

            if not os.path.exists(abs_path):
                return {"error": f"File not found: {path}"}

            if not os.path.isfile(abs_path):
                return {"error": f"Not a file: {path}"}

            # Check size
            size = os.path.getsize(abs_path)
            if size > max_size_mb * 1024 * 1024:
                return {"error": f"File too large: {size / (1024*1024):.2f} MB (max: {max_size_mb} MB)"}

            with open(abs_path, "r", encoding=encoding) as f:
                content = f.read()

            return {
                "path": abs_path,
                "size": size,
                "content": content,
                "lines": len(content.splitlines()),
            }

        except UnicodeDecodeError:
            return {"error": f"Cannot decode file as {encoding} (might be binary)"}
        except Exception as e:
            return {"error": str(e)}

    def read_file_lines(
        self,
        path: str,
        start_line: int = 1,
        num_lines: int = 100,
    ) -> Dict[str, Any]:
        """Read specific lines from a file."""
        try:
            abs_path = os.path.abspath(path)

            if not self._is_path_allowed(abs_path):
                return {"error": f"Path not allowed: {path}"}

            if not os.path.exists(abs_path):
                return {"error": f"File not found: {path}"}

            with open(abs_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)
            start_idx = max(0, start_line - 1)
            end_idx = min(total_lines, start_idx + num_lines)

            lines = all_lines[start_idx:end_idx]

            return {
                "path": abs_path,
                "total_lines": total_lines,
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "lines": [{"num": start_idx + i + 1, "content": line.rstrip()} for i, line in enumerate(lines)],
            }

        except Exception as e:
            return {"error": str(e)}

    def list_directory(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False,
        include_hidden: bool = False,
    ) -> Dict[str, Any]:
        """
        List directory contents.

        Args:
            path: Directory path
            pattern: Glob pattern
            recursive: Include subdirectories
            include_hidden: Include hidden files

        Returns:
            Dict with file listing
        """
        try:
            abs_path = os.path.abspath(path)

            if not self._is_path_allowed(abs_path):
                return {"error": f"Path not allowed: {path}"}

            if not os.path.exists(abs_path):
                return {"error": f"Directory not found: {path}"}

            if not os.path.isdir(abs_path):
                return {"error": f"Not a directory: {path}"}

            if recursive:
                search_pattern = os.path.join(abs_path, "**", pattern)
                files = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(abs_path, pattern)
                files = glob.glob(search_pattern)

            entries = []
            for f in files:
                if not include_hidden and os.path.basename(f).startswith("."):
                    continue

                try:
                    stat = os.stat(f)
                    entries.append({
                        "name": os.path.basename(f),
                        "path": f,
                        "is_dir": os.path.isdir(f),
                        "size": stat.st_size if os.path.isfile(f) else None,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                except OSError:
                    continue

            # Sort: directories first, then by name
            entries.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

            return {
                "path": abs_path,
                "pattern": pattern,
                "count": len(entries),
                "entries": entries[:1000],  # Limit to 1000 entries
            }

        except Exception as e:
            return {"error": str(e)}

    def search_files(
        self,
        path: str,
        pattern: str,
        content_pattern: Optional[str] = None,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Search for files by name pattern and optionally content.

        Args:
            path: Base directory
            pattern: File name pattern (glob)
            content_pattern: Optional text to search in files
            max_results: Maximum results

        Returns:
            Dict with matching files
        """
        try:
            abs_path = os.path.abspath(path)

            if not self._is_path_allowed(abs_path):
                return {"error": f"Path not allowed: {path}"}

            search_pattern = os.path.join(abs_path, "**", pattern)
            files = glob.glob(search_pattern, recursive=True)

            matches = []
            for f in files[:max_results * 10]:  # Search more, filter down
                if len(matches) >= max_results:
                    break

                if not os.path.isfile(f):
                    continue

                match = {
                    "path": f,
                    "name": os.path.basename(f),
                    "size": os.path.getsize(f),
                }

                # If content search requested
                if content_pattern:
                    try:
                        with open(f, "r", encoding="utf-8", errors="ignore") as file:
                            content = file.read()
                            if content_pattern.lower() in content.lower():
                                # Find line numbers
                                lines_with_match = []
                                for i, line in enumerate(content.splitlines(), 1):
                                    if content_pattern.lower() in line.lower():
                                        lines_with_match.append({
                                            "line": i,
                                            "content": line.strip()[:200],
                                        })
                                match["matches"] = lines_with_match[:10]
                                matches.append(match)
                    except Exception:
                        continue
                else:
                    matches.append(match)

            return {
                "path": abs_path,
                "pattern": pattern,
                "content_pattern": content_pattern,
                "count": len(matches),
                "results": matches,
            }

        except Exception as e:
            return {"error": str(e)}

    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed file information."""
        try:
            abs_path = os.path.abspath(path)

            if not self._is_path_allowed(abs_path):
                return {"error": f"Path not allowed: {path}"}

            if not os.path.exists(abs_path):
                return {"error": f"Path not found: {path}"}

            stat = os.stat(abs_path)

            return {
                "path": abs_path,
                "name": os.path.basename(abs_path),
                "is_file": os.path.isfile(abs_path),
                "is_dir": os.path.isdir(abs_path),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
            }

        except Exception as e:
            return {"error": str(e)}

    # ==================== Command Execution ====================

    def _is_command_safe(self, command: str) -> bool:
        """
        Check if command is safe to execute using pattern matching.

        This provides defense-in-depth against command injection by checking
        for dangerous patterns that could harm the system.
        """
        command_lower = command.lower()

        # Check blocked commands from configuration
        for blocked in self.blocked_commands:
            if blocked.lower() in command_lower:
                logger.warning(f"Blocked command pattern detected: {blocked}")
                return False

        # Check against dangerous regex patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                logger.warning(f"Dangerous command pattern detected: {pattern}")
                return False

        # Check for shell metacharacters that could enable injection
        # All pipe, chaining, and substitution characters are blocked unconditionally
        dangerous_chars = ['|', ';', '`', '$', '\n', '\r']
        for char in dangerous_chars:
            if char in command:
                logger.warning(f"Dangerous character '{char}' in command")
                return False

        # Block '&' but allow '&&' for conditional chaining
        if '&' in command and '&&' not in command:
            logger.warning("Background execution '&' detected")
            return False

        return True

    async def execute_command(
        self,
        command: Union[str, List[str]],
        cwd: Optional[str] = None,
        timeout: int = 60,
        use_shell: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a command safely.

        Args:
            command: Command to execute (string or list of args)
            cwd: Working directory
            timeout: Timeout in seconds
            use_shell: Use shell execution (DISCOURAGED - security risk)

        Returns:
            Dict with output or error

        Security Notes:
            - Prefer passing command as a list of arguments (use_shell=False)
            - Shell mode is discouraged and applies extra safety checks
            - All commands are validated against dangerous patterns
        """
        if not self.enable_command_execution:
            return {"error": "Command execution is disabled"}

        # Convert to string for safety check
        command_str = command if isinstance(command, str) else ' '.join(command)

        if not self._is_command_safe(command_str):
            return {"error": "Command blocked for safety reasons"}

        try:
            # Validate cwd if provided
            if cwd:
                if not self._is_path_allowed(cwd):
                    return {"error": f"Working directory not allowed: {cwd}"}
                cwd_path = Path(cwd).resolve()
                if not cwd_path.is_dir():
                    return {"error": f"Working directory not found: {cwd}"}
                cwd = str(cwd_path)

            if use_shell:
                # Shell mode - use with caution
                logger.warning(f"Executing command in shell mode: {command_str[:50]}...")
                process = await asyncio.create_subprocess_shell(
                    command_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            else:
                # Safer non-shell mode
                if isinstance(command, str):
                    # Parse string into args safely
                    try:
                        args = shlex.split(command)
                    except ValueError as e:
                        return {"error": f"Invalid command syntax: {e}"}
                else:
                    args = command

                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()  # Clean up zombie process
                return {"error": f"Command timed out after {timeout}s"}

            return {
                "command": command_str,
                "cwd": cwd or os.getcwd(),
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "success": process.returncode == 0,
            }

        except FileNotFoundError:
            return {"error": f"Command not found: {command_str.split()[0] if command_str else 'empty'}"}
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {"error": f"Execution error: {type(e).__name__}"}

    def execute_command_sync(
        self,
        command: Union[str, List[str]],
        cwd: Optional[str] = None,
        timeout: int = 60,
        use_shell: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute command synchronously and safely.

        Args:
            command: Command to execute (string or list of args)
            cwd: Working directory
            timeout: Timeout in seconds
            use_shell: Use shell execution (DISCOURAGED)

        Returns:
            Dict with output or error
        """
        if not self.enable_command_execution:
            return {"error": "Command execution is disabled"}

        # Convert to string for safety check
        command_str = command if isinstance(command, str) else ' '.join(command)

        if not self._is_command_safe(command_str):
            return {"error": "Command blocked for safety reasons"}

        try:
            # Validate cwd
            if cwd:
                if not self._is_path_allowed(cwd):
                    return {"error": f"Working directory not allowed: {cwd}"}
                cwd_path = Path(cwd).resolve()
                if not cwd_path.is_dir():
                    return {"error": f"Working directory not found: {cwd}"}
                cwd = str(cwd_path)

            if use_shell:
                logger.warning(f"Executing command in shell mode: {command_str[:50]}...")
                result = subprocess.run(
                    command_str,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=cwd,
                )
            else:
                # Parse command safely
                if isinstance(command, str):
                    try:
                        args = shlex.split(command)
                    except ValueError as e:
                        return {"error": f"Invalid command syntax: {e}"}
                else:
                    args = command

                result = subprocess.run(
                    args,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=cwd,
                )

            return {
                "command": command_str,
                "cwd": cwd or os.getcwd(),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout}s"}
        except FileNotFoundError:
            return {"error": f"Command not found: {command_str.split()[0] if command_str else 'empty'}"}
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {"error": f"Execution error: {type(e).__name__}"}

    # ==================== Environment ====================

    def get_environment_variables(self, filter_pattern: Optional[str] = None) -> Dict[str, str]:
        """Get environment variables."""
        env_vars = dict(os.environ)

        if filter_pattern:
            filter_lower = filter_pattern.lower()
            env_vars = {k: v for k, v in env_vars.items() if filter_lower in k.lower()}

        # Mask sensitive values
        sensitive_keys = ["key", "secret", "password", "token", "credential"]
        masked_vars = {}
        for k, v in env_vars.items():
            if any(s in k.lower() for s in sensitive_keys):
                masked_vars[k] = v[:4] + "..." + v[-4:] if len(v) > 8 else "***"
            else:
                masked_vars[k] = v

        return masked_vars

    def get_installed_packages(self) -> List[Dict[str, str]]:
        """Get list of installed Python packages."""
        try:
            import pkg_resources
            packages = []
            for pkg in pkg_resources.working_set:
                packages.append({
                    "name": pkg.project_name,
                    "version": pkg.version,
                    "location": pkg.location,
                })
            packages.sort(key=lambda x: x["name"].lower())
            return packages
        except Exception as e:
            return [{"error": str(e)}]

    def get_python_info(self) -> Dict[str, Any]:
        """Get Python environment information."""
        return {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
            },
            "executable": sys.executable,
            "platform": sys.platform,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "is_virtualenv": sys.prefix != sys.base_prefix,
            "path": sys.path[:10],
        }
