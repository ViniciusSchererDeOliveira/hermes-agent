#!/usr/bin/env python3
"""
Dataset Logger for Hermes Agent.

Provides continuous logging of conversation exchanges to JSONL files
for fine-tuning and auditing purposes.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


class DatasetLogger:
    """
    Logs conversation exchanges to JSONL files for dataset creation.
    
    Each log entry represents a complete user->assistant turn including
    tool usage and metadata.
    """
    
    def __init__(self, session_id: str):
        """
        Initialize the dataset logger for a specific session.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        hermes_home = get_hermes_home()
        self.log_dir = hermes_home / "datasets"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"dataset_{session_id}.jsonl"
        
    def log_exchange(self, 
                    session_id: str,
                    model_used: str,
                    system_prompt: str,
                    messages: list,
                    metadata: dict):
        """
        Log a complete exchange (user->assistant turn) to JSONL file.
        
        Args:
            session_id: Unique session identifier
            model_used: Model name used for this exchange
            system_prompt: System prompt active during exchange
            messages: List of message dicts representing the exchange
            metadata: Dict with token counts, timing, etc.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "model_used": model_used,
            "system_prompt": system_prompt,
            "messages": messages,
            "metadata": metadata
        }
        
        # Append to JSONL file with error resilience
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write dataset log entry: {e}")