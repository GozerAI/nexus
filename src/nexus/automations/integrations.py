"""
Integrations - Direct connectors for external services
"""

import os
import asyncio
import json
import base64
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time

from .triggers import TriggerManager, TriggerType, TriggerEvent


# ============================================================
# Gmail Connector
# ============================================================

@dataclass
class GmailConfig:
    """Gmail connector configuration."""
    credentials_path: str = ""
    token_path: str = ""
    check_interval_minutes: int = 5
    max_results: int = 20
    label_filter: str = "INBOX"
    unread_only: bool = True


@dataclass
class EmailMessage:
    """Represents a Gmail message."""
    id: str
    thread_id: str
    sender: str
    subject: str
    body: str
    received_at: datetime
    labels: List[str] = field(default_factory=list)
    has_attachments: bool = False
    snippet: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "sender": self.sender,
            "subject": self.subject,
            "body": self.body[:1000],
            "received_at": self.received_at.isoformat(),
            "labels": self.labels,
            "has_attachments": self.has_attachments,
        }


class GmailConnector:
    """
    Gmail integration for email automation.
    
    Requires Google API credentials. See:
    https://developers.google.com/gmail/api/quickstart/python
    
    Usage:
        connector = GmailConnector(config, trigger_manager)
        await connector.start_watching()
    """
    
    def __init__(self, config: GmailConfig, triggers: TriggerManager = None):
        self.config = config
        self.triggers = triggers
        self._service = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_check: Optional[datetime] = None
        self._processed_ids: set = set()
    
    def is_configured(self) -> bool:
        """Check if Gmail API is configured."""
        return (
            self.config.credentials_path and 
            os.path.exists(self.config.credentials_path)
        )
    
    async def initialize(self) -> bool:
        """Initialize Gmail API service."""
        if not self.is_configured():
            print("Gmail not configured - provide credentials_path")
            return False
        
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            
            SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
            
            creds = None
            if os.path.exists(self.config.token_path):
                creds = Credentials.from_authorized_user_file(
                    self.config.token_path, SCOPES
                )
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.config.credentials_path, SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                
                with open(self.config.token_path, 'w') as token:
                    token.write(creds.to_json())
            
            self._service = build('gmail', 'v1', credentials=creds)
            return True
            
        except ImportError:
            print("Gmail API libraries not installed. Run:")
            print("pip install google-auth-oauthlib google-api-python-client")
            return False
        except Exception as e:
            print(f"Gmail initialization failed: {e}")
            return False
    
    async def fetch_emails(self, max_results: int = None) -> List[EmailMessage]:
        """Fetch recent emails."""
        if not self._service:
            return []
        
        max_results = max_results or self.config.max_results
        
        try:
            query = f"in:{self.config.label_filter}"
            if self.config.unread_only:
                query += " is:unread"
            
            results = self._service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for msg_ref in messages:
                msg = self._service.users().messages().get(
                    userId='me',
                    id=msg_ref['id'],
                    format='full'
                ).execute()
                
                email = self._parse_message(msg)
                if email:
                    emails.append(email)
            
            return emails
            
        except Exception as e:
            print(f"Gmail fetch failed: {e}")
            return []
    
    def _parse_message(self, msg: Dict) -> Optional[EmailMessage]:
        """Parse Gmail API message to EmailMessage."""
        try:
            headers = {h['name']: h['value'] for h in msg['payload']['headers']}
            
            # Get body
            body = ""
            if 'parts' in msg['payload']:
                for part in msg['payload']['parts']:
                    if part['mimeType'] == 'text/plain':
                        data = part['body'].get('data', '')
                        body = base64.urlsafe_b64decode(data).decode('utf-8')
                        break
            elif 'body' in msg['payload']:
                data = msg['payload']['body'].get('data', '')
                if data:
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
            
            # Parse date
            date_str = headers.get('Date', '')
            try:
                from email.utils import parsedate_to_datetime
                received = parsedate_to_datetime(date_str)
            except (TypeError, ValueError) as e:
                # Invalid or missing date format
                received = datetime.now()
            
            return EmailMessage(
                id=msg['id'],
                thread_id=msg['threadId'],
                sender=headers.get('From', ''),
                subject=headers.get('Subject', ''),
                body=body,
                received_at=received,
                labels=msg.get('labelIds', []),
                snippet=msg.get('snippet', ''),
                has_attachments='parts' in msg['payload'] and len(msg['payload']['parts']) > 1
            )
        except Exception as e:
            print(f"Message parse error: {e}")
            return None
    
    def start_watching(self):
        """Start background email watching."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
    
    def stop_watching(self):
        """Stop watching for emails."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _watch_loop(self):
        """Background loop checking for new emails."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self._running:
            try:
                emails = loop.run_until_complete(self.fetch_emails())
                
                for email in emails:
                    if email.id not in self._processed_ids:
                        self._processed_ids.add(email.id)
                        self._fire_email_event(email)
                
                # Limit processed IDs memory
                if len(self._processed_ids) > 1000:
                    self._processed_ids = set(list(self._processed_ids)[-500:])
                
                self._last_check = datetime.now()
                
            except Exception as e:
                print(f"Gmail watch error: {e}")
            
            time.sleep(self.config.check_interval_minutes * 60)
        
        loop.close()
    
    def _fire_email_event(self, email: EmailMessage):
        """Fire trigger event for new email."""
        if self.triggers:
            self.triggers.fire(
                TriggerType.EMAIL_RECEIVED,
                data=email.to_dict(),
                source="gmail"
            )


# ============================================================
# Notion Connector
# ============================================================

@dataclass
class NotionConfig:
    """Notion connector configuration."""
    api_key: str = ""
    approval_database_id: str = ""
    check_interval_minutes: int = 2


@dataclass
class ApprovalItem:
    """An approval request in Notion."""
    id: str
    title: str
    status: str  # pending, approved, rejected
    pipeline_id: str
    step_id: str
    details: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class NotionConnector:
    """
    Notion integration for approval workflows.
    
    Creates approval requests as Notion pages, watches for status changes.
    
    Usage:
        connector = NotionConnector(config, trigger_manager)
        await connector.create_approval(request)
    """
    
    def __init__(self, config: NotionConfig, triggers: TriggerManager = None):
        self.config = config
        self.triggers = triggers
        self._client = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pending_approvals: Dict[str, ApprovalItem] = {}
    
    def is_configured(self) -> bool:
        """Check if Notion API is configured."""
        return bool(self.config.api_key and self.config.approval_database_id)
    
    async def initialize(self) -> bool:
        """Initialize Notion client."""
        if not self.is_configured():
            print("Notion not configured - provide api_key and approval_database_id")
            return False
        
        try:
            from notion_client import Client
            self._client = Client(auth=self.config.api_key)
            return True
        except ImportError:
            print("Notion client not installed. Run: pip install notion-client")
            return False
        except Exception as e:
            print(f"Notion initialization failed: {e}")
            return False
    
    async def create_approval(
        self,
        title: str,
        pipeline_id: str,
        step_id: str,
        details: Dict[str, Any]
    ) -> Optional[str]:
        """Create an approval request in Notion."""
        if not self._client:
            return None
        
        try:
            page = self._client.pages.create(
                parent={"database_id": self.config.approval_database_id},
                properties={
                    "Name": {"title": [{"text": {"content": title}}]},
                    "Status": {"select": {"name": "Pending"}},
                    "Pipeline": {"rich_text": [{"text": {"content": pipeline_id}}]},
                    "Step": {"rich_text": [{"text": {"content": step_id}}]},
                },
                children=[
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"text": {"content": json.dumps(details, indent=2)}}]
                        }
                    }
                ]
            )
            
            approval = ApprovalItem(
                id=page["id"],
                title=title,
                status="pending",
                pipeline_id=pipeline_id,
                step_id=step_id,
                details=details,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self._pending_approvals[page["id"]] = approval
            
            return page["id"]
            
        except Exception as e:
            print(f"Notion create failed: {e}")
            return None
    
    async def check_approvals(self) -> List[ApprovalItem]:
        """Check for approval status changes."""
        if not self._client:
            return []
        
        updated = []
        
        try:
            for page_id, approval in list(self._pending_approvals.items()):
                page = self._client.pages.retrieve(page_id)
                
                status_prop = page["properties"].get("Status", {})
                new_status = status_prop.get("select", {}).get("name", "").lower()
                
                if new_status != approval.status:
                    approval.status = new_status
                    approval.updated_at = datetime.now()
                    updated.append(approval)
                    
                    if new_status in ["approved", "rejected"]:
                        del self._pending_approvals[page_id]
            
            return updated
            
        except Exception as e:
            print(f"Notion check failed: {e}")
            return []
    
    def start_watching(self):
        """Start watching for approval changes."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
    
    def stop_watching(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _watch_loop(self):
        """Background loop checking approvals."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self._running:
            try:
                updated = loop.run_until_complete(self.check_approvals())
                
                for approval in updated:
                    if self.triggers:
                        trigger_type = TriggerType.WEBHOOK  # Using webhook as approval event
                        self.triggers.fire(
                            trigger_type,
                            data={
                                "approval_id": approval.id,
                                "status": approval.status,
                                "pipeline_id": approval.pipeline_id,
                                "step_id": approval.step_id,
                            },
                            source="notion"
                        )
                
            except Exception as e:
                print(f"Notion watch error: {e}")
            
            time.sleep(self.config.check_interval_minutes * 60)
        
        loop.close()


# ============================================================
# File Watcher
# ============================================================

@dataclass
class FileWatcherConfig:
    """File watcher configuration."""
    watch_paths: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=lambda: ["*.*"])
    recursive: bool = True
    debounce_seconds: float = 1.0


class FileWatcher:
    """
    File system watcher for triggering automations on file changes.
    
    Watches directories for new/modified files and fires triggers.
    
    Usage:
        watcher = FileWatcher(config, trigger_manager)
        watcher.start_watching()
    """
    
    def __init__(self, config: FileWatcherConfig, triggers: TriggerManager = None):
        self.config = config
        self.triggers = triggers
        self._running = False
        self._observer = None
        self._last_events: Dict[str, datetime] = {}
    
    def start_watching(self):
        """Start file watching."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class Handler(FileSystemEventHandler):
                def __init__(self, watcher):
                    self.watcher = watcher
                
                def on_created(self, event):
                    if not event.is_directory:
                        self.watcher._handle_event("created", event.src_path)
                
                def on_modified(self, event):
                    if not event.is_directory:
                        self.watcher._handle_event("modified", event.src_path)
            
            self._observer = Observer()
            handler = Handler(self)
            
            for path in self.config.watch_paths:
                if os.path.exists(path):
                    self._observer.schedule(
                        handler, path, recursive=self.config.recursive
                    )
            
            self._running = True
            self._observer.start()
            
        except ImportError:
            print("watchdog not installed. Run: pip install watchdog")
        except Exception as e:
            print(f"FileWatcher start failed: {e}")
    
    def stop_watching(self):
        """Stop file watching."""
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join()
    
    def _handle_event(self, event_type: str, file_path: str):
        """Handle file system event with debouncing."""
        now = datetime.now()
        
        # Debounce
        last = self._last_events.get(file_path)
        if last and (now - last).total_seconds() < self.config.debounce_seconds:
            return
        
        self._last_events[file_path] = now
        
        # Check patterns
        import fnmatch
        filename = os.path.basename(file_path)
        matches = any(fnmatch.fnmatch(filename, p) for p in self.config.patterns)
        
        if not matches:
            return
        
        # Fire trigger
        if self.triggers:
            trigger_type = (
                TriggerType.FILE_CREATED if event_type == "created"
                else TriggerType.FILE_MODIFIED
            )
            self.triggers.fire(
                trigger_type,
                data={
                    "path": file_path,
                    "filename": filename,
                    "event": event_type,
                },
                source="filesystem"
            )
    
    def is_running(self) -> bool:
        return self._running
