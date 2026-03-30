"""
Smart Automation Router - Folder-based and AI-powered routing

Instead of relying on file extensions, this uses:
1. Folder-based routing - Drop files in specific folders
2. Smart inbox - AI determines what to do with any file
3. Command queue - Explicit instructions via command file

Folder Structure:
    ~/Automations/
    ├── Inbox/              → Smart routing (AI decides)
    ├── Ebooks/             → Ebook generation
    ├── Research/           → Research reports
    ├── Meetings/           → Meeting notes extraction
    ├── Repurpose/          → Content repurposing
    ├── Proposals/          → Proposal generation
    ├── Commands/           → Command queue
    └── Output/             → All outputs
"""

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum

from .triggers import TriggerType, TriggerEvent, TriggerManager


class WorkflowType(Enum):
    """Available workflow types."""
    EBOOK = "ebook"
    RESEARCH = "research"
    MEETING_NOTES = "meeting_notes"
    REPURPOSE = "repurpose"
    PROPOSAL = "proposal"
    DIGEST = "digest"
    EMAIL_TRIAGE = "email_triage"
    UNKNOWN = "unknown"


@dataclass
class ProcessingJob:
    """A job to be processed."""
    id: str
    workflow_type: WorkflowType
    source_file: Path
    inputs: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    output_path: Optional[Path] = None
    error: Optional[str] = None


class SmartRouter:
    """
    Intelligent file router that determines what to do with uploaded files.
    
    Routes based on:
    1. Which folder the file was placed in
    2. File content analysis (for inbox)
    3. Explicit commands
    """
    
    # Folder to workflow mapping
    FOLDER_ROUTES = {
        "Ebooks": WorkflowType.EBOOK,
        "Research": WorkflowType.RESEARCH,
        "Meetings": WorkflowType.MEETING_NOTES,
        "Repurpose": WorkflowType.REPURPOSE,
        "Proposals": WorkflowType.PROPOSAL,
    }
    
    def __init__(self, platform, pipelines: Dict[str, Any], base_path: Path = None):
        self.platform = platform
        self.pipelines = pipelines
        self.base_path = base_path or (Path.home() / "Automations")
        self.output_path = self.base_path / "Output"
        self.inbox_path = self.base_path / "Inbox"
        self.commands_path = self.base_path / "Commands"
        
        self._jobs: List[ProcessingJob] = []
        self._processing = False
    
    def setup_folders(self):
        """Create the folder structure."""
        folders = [
            self.base_path,
            self.output_path,
            self.inbox_path,
            self.commands_path,
            self.base_path / "Ebooks",
            self.base_path / "Research",
            self.base_path / "Meetings",
            self.base_path / "Repurpose",
            self.base_path / "Proposals",
            self.base_path / "Processed",  # Archive of processed files
        ]
        
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Create README in base folder
        readme_path = self.base_path / "README.txt"
        if not readme_path.exists():
            readme_path.write_text(self._get_readme())
        
        return folders
    
    def _get_readme(self) -> str:
        return """
AUTOMATION FOLDERS
==================

Drop files in these folders to trigger automations:

📁 Inbox/
   Smart routing - AI analyzes the file and decides what to do.
   Good for: Any file when you're not sure which workflow to use.

📁 Ebooks/
   Generate ebooks from content briefs.
   Input: Text file with topic on line 1, audience on line 2.
   Output: Markdown ebook in Output/

📁 Research/
   Generate comprehensive research reports.
   Input: Text file with topic on line 1, questions on following lines.
   Output: Research report in Output/

📁 Meetings/
   Extract action items and notes from meeting transcripts.
   Input: Text file with meeting transcript.
   Output: Structured notes + action items in Output/

📁 Repurpose/
   Transform content into multiple formats.
   Input: Any article, blog post, or content file.
   Output: Blog post, LinkedIn, Twitter thread, email newsletter.

📁 Proposals/
   Generate professional proposals.
   Input: Text file with client, problem, solution details.
   Output: Full proposal + cover letter + one-pager.

📁 Commands/
   Explicit command queue. Create a .json file with:
   {
       "workflow": "research",
       "inputs": {
           "topic": "AI Market Trends",
           "questions": ["What are growth projections?"]
       }
   }

📁 Output/
   All generated outputs appear here.

📁 Processed/
   Archive of processed input files.

"""
    
    def route_file(self, file_path: Path) -> WorkflowType:
        """Determine which workflow to use based on file location."""
        # Check which folder the file is in
        for folder_name, workflow_type in self.FOLDER_ROUTES.items():
            folder_path = self.base_path / folder_name
            if file_path.parent == folder_path:
                return workflow_type
        
        # Check if it's in inbox (needs smart routing)
        if file_path.parent == self.inbox_path:
            return self._smart_route(file_path)
        
        # Check if it's a command file
        if file_path.parent == self.commands_path:
            return self._parse_command(file_path)
        
        return WorkflowType.UNKNOWN
    
    def _smart_route(self, file_path: Path) -> WorkflowType:
        """Use AI to determine what to do with an inbox file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')[:3000]
        except (IOError, OSError, UnicodeDecodeError) as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not read file {file_path}: {e}")
            return WorkflowType.UNKNOWN
        
        # Simple heuristics first (fast)
        content_lower = content.lower()
        
        # Meeting detection
        meeting_signals = ['transcript', 'meeting', 'attendees', 'action items', 
                         'minutes', 'discussed', 'agreed', 'next steps']
        if sum(1 for s in meeting_signals if s in content_lower) >= 2:
            return WorkflowType.MEETING_NOTES
        
        # Proposal detection
        proposal_signals = ['client', 'proposal', 'solution', 'problem', 'budget', 
                          'timeline', 'deliverables', 'scope']
        if sum(1 for s in proposal_signals if s in content_lower) >= 3:
            return WorkflowType.PROPOSAL
        
        # Research detection
        research_signals = ['research', 'analyze', 'investigate', 'questions', 
                          'market', 'trends', 'data', 'findings']
        if sum(1 for s in research_signals if s in content_lower) >= 2:
            return WorkflowType.RESEARCH
        
        # Content repurpose detection (article-like content)
        # Long content with paragraphs or multiple sentences
        if len(content) > 800:
            # Has paragraph breaks or is substantial prose
            if '\n\n' in content or content.count('. ') > 10:
                return WorkflowType.REPURPOSE
        
        # Default to ebook for structured briefs
        if len(content.split('\n')) <= 10:
            return WorkflowType.EBOOK
        
        return WorkflowType.UNKNOWN
    
    def _parse_command(self, file_path: Path) -> WorkflowType:
        """Parse a command file."""
        try:
            if file_path.suffix == '.json':
                data = json.loads(file_path.read_text())
                workflow = data.get('workflow', '').lower()
                return WorkflowType(workflow) if workflow in [w.value for w in WorkflowType] else WorkflowType.UNKNOWN
        except (IOError, json.JSONDecodeError, ValueError) as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not parse command file {file_path}: {e}")
        return WorkflowType.UNKNOWN
    
    def parse_file_inputs(self, file_path: Path, workflow_type: WorkflowType) -> Dict[str, Any]:
        """Parse file content into workflow inputs."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except (IOError, OSError) as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not read file {file_path}: {e}")
            return {}
        
        lines = content.strip().split('\n')
        
        if workflow_type == WorkflowType.EBOOK:
            return {
                "topic": lines[0] if lines else file_path.stem,
                "audience": lines[1] if len(lines) > 1 else "general readers",
                "tone": lines[2] if len(lines) > 2 else "professional"
            }
        
        elif workflow_type == WorkflowType.RESEARCH:
            return {
                "topic": lines[0] if lines else file_path.stem,
                "context": lines[1] if len(lines) > 1 else "",
                "questions": lines[2:] if len(lines) > 2 else []
            }
        
        elif workflow_type == WorkflowType.MEETING_NOTES:
            return {
                "transcript": content,
                "meeting_title": file_path.stem.replace('_', ' ').replace('-', ' ').title(),
                "attendees": [],
                "meeting_date": datetime.now().strftime("%Y-%m-%d")
            }
        
        elif workflow_type == WorkflowType.REPURPOSE:
            return {
                "source_content": content,
                "content_type": "article",
                "key_message": ""
            }
        
        elif workflow_type == WorkflowType.PROPOSAL:
            # Try to parse structured format
            inputs = {
                "client_name": "",
                "project_name": file_path.stem,
                "problem": "",
                "solution_brief": "",
                "differentiators": []
            }
            
            current_field = None
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('-'):
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'client' in key:
                        inputs['client_name'] = value
                    elif 'project' in key:
                        inputs['project_name'] = value
                    elif 'problem' in key:
                        inputs['problem'] = value
                        current_field = 'problem'
                    elif 'solution' in key:
                        inputs['solution_brief'] = value
                        current_field = 'solution'
                    elif 'diff' in key:
                        current_field = 'differentiators'
                elif line.startswith('- ') and current_field == 'differentiators':
                    inputs['differentiators'].append(line[2:])
                elif current_field in ['problem', 'solution'] and line:
                    inputs[current_field + '_brief' if current_field == 'solution' else current_field] += ' ' + line
            
            # Fallback if not structured
            if not inputs['problem'] and not inputs['client_name']:
                inputs['client_name'] = lines[0] if lines else "Client"
                inputs['problem'] = lines[1] if len(lines) > 1 else content[:500]
                inputs['solution_brief'] = lines[2] if len(lines) > 2 else ""
            
            return inputs
        
        # Command file
        if file_path.suffix == '.json':
            try:
                data = json.loads(content)
                return data.get('inputs', {})
            except json.JSONDecodeError as e:
                import logging
                logging.getLogger(__name__).debug(f"Could not parse JSON inputs from {file_path}: {e}")

        return {"content": content}
    
    async def process_file(self, file_path: Path) -> ProcessingJob:
        """Process a single file."""
        job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_path.stem[:20]}"
        
        # Route the file
        workflow_type = self.route_file(file_path)
        
        if workflow_type == WorkflowType.UNKNOWN:
            return ProcessingJob(
                id=job_id,
                workflow_type=workflow_type,
                source_file=file_path,
                inputs={},
                status="error",
                error="Could not determine workflow type. Move to a specific folder."
            )
        
        # Parse inputs
        inputs = self.parse_file_inputs(file_path, workflow_type)
        
        job = ProcessingJob(
            id=job_id,
            workflow_type=workflow_type,
            source_file=file_path,
            inputs=inputs
        )
        
        # Get the pipeline
        pipeline = self.pipelines.get(workflow_type.value)
        if not pipeline:
            job.status = "error"
            job.error = f"Pipeline not configured for {workflow_type.value}"
            return job
        
        # Run the pipeline
        try:
            job.status = "running"
            result = await pipeline.run(inputs, run_id=job_id)
            
            # Save output
            output_file = self.output_path / f"{job_id}_{workflow_type.value}.md"
            
            # Compile outputs to markdown
            output_content = f"# {workflow_type.value.replace('_', ' ').title()}\n\n"
            output_content += f"**Source:** {file_path.name}\n"
            output_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            output_content += "---\n\n"
            
            if result.outputs:
                for key, value in result.outputs.items():
                    output_content += f"## {key.replace('_', ' ').title()}\n\n"
                    if isinstance(value, str):
                        output_content += value + "\n\n"
                    elif isinstance(value, dict):
                        output_content += json.dumps(value, indent=2) + "\n\n"
                    else:
                        output_content += str(value) + "\n\n"
            
            output_file.write_text(output_content, encoding='utf-8')
            
            job.status = "completed"
            job.output_path = output_file
            
            # Archive the source file
            archive_path = self.base_path / "Processed" / file_path.name
            shutil.move(str(file_path), str(archive_path))
            
        except Exception as e:
            job.status = "error"
            job.error = str(e)
        
        self._jobs.append(job)
        return job
    
    def get_pending_files(self) -> List[Path]:
        """Get all files waiting to be processed."""
        files = []
        
        # Check all input folders
        for folder_name in list(self.FOLDER_ROUTES.keys()) + ["Inbox", "Commands"]:
            folder = self.base_path / folder_name
            if folder.exists():
                for file in folder.iterdir():
                    if file.is_file() and not file.name.startswith('.'):
                        files.append(file)
        
        return files
    
    def get_jobs(self, limit: int = 50) -> List[ProcessingJob]:
        """Get recent jobs."""
        return self._jobs[-limit:]
