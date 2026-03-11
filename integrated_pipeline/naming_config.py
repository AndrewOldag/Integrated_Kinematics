"""Naming template for batch output CSV files compatible with the kinematics viewer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class NamingConfig:
    enabled: bool = False
    delimiter: str = "_"          # splits the input folder name
    scope_code: str = "k"         # "k" or "p"
    condition_token_index: int = 0
    time_token_index: int = 1
    replicate_token_index: int = 2
    strip_non_numeric_from_time: bool = True
    output_subdir: str = "analysis_ready"  # flat dir inside output_root

    def derive_name(self, folder_name: str) -> Optional[str]:
        """Derive a kinematics-viewer-compatible CSV stem from a folder name.

        Returns None if the folder name does not have enough tokens.
        """
        parts = folder_name.split(self.delimiter)
        required = max(
            self.condition_token_index,
            self.time_token_index,
            self.replicate_token_index,
        ) + 1
        if len(parts) < required:
            return None
        cond = parts[self.condition_token_index]
        time_raw = parts[self.time_token_index]
        repl = parts[self.replicate_token_index]
        time_str = re.sub(r"[^0-9.]", "", time_raw) if self.strip_non_numeric_from_time else time_raw
        return f"{self.scope_code}_{cond}_{time_str or time_raw}_{repl}"
