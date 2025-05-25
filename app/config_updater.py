import json
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict

from .config import RUNTIME_CONFIG_PATH

# `fcntl` provides POSIX file locking. It is unavailable on Windows. When not
# present we simply skip locking which is safe for single-user scenarios.
try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - not available on Windows
    fcntl = None  # type: ignore


class ConfigUpdater:
    def __init__(self, config_path: str = RUNTIME_CONFIG_PATH):
        self.config_path = config_path

    def _read_config(self) -> Dict:
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, "r") as f:
            if fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            finally:
                if fcntl:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data

    def _write_config(self, config: Dict) -> None:
        dir_name = os.path.dirname(self.config_path)
        os.makedirs(dir_name, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
            json.dump(config, tf, indent=2)
            temp_name = tf.name
        
        # Try atomic replace first
        try:
            os.replace(temp_name, self.config_path)
        except OSError as e:
            # If atomic replace fails (common in Docker with mounted volumes),
            # fall back to copy and delete
            import shutil
            try:
                # Make a backup first
                if os.path.exists(self.config_path):
                    backup_path = f"{self.config_path}.backup"
                    shutil.copy2(self.config_path, backup_path)
                
                # Copy the temp file to the target
                shutil.copy2(temp_name, self.config_path)
                
                # Remove the temp file
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass  # Ignore errors removing temp file
                    
                # Remove backup if successful
                if os.path.exists(backup_path):
                    try:
                        os.unlink(backup_path)
                    except OSError:
                        pass
            except Exception:
                # If anything goes wrong, try to restore backup
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, self.config_path)
                raise

    def get_current_ontology_version(self, ontology_name: str) -> Optional[str]:
        config = self._read_config()
        entry = config.get(ontology_name)
        if entry:
            return entry.get("collection")
        return None

    def update_ontology_version(
        self, ontology_name: str, new_collection_name: str, source_url: str
    ) -> None:
        config = self._read_config()
        config[ontology_name] = {
            "collection": new_collection_name,
            "source_url": source_url,
            "last_updated": datetime.now().isoformat(),
        }
        self._write_config(config)

    def get_all_ontology_configs(self) -> Dict:
        return self._read_config()
