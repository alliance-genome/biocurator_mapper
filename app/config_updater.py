import json
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, List

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


class DownloadHistoryManager:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.environ.get("ONTOLOGY_DATA_DIR", "/app/data")
        self.history_file = os.path.join(data_dir, "ontology_downloads_history.json")
        self.max_records_per_ontology = 10

    def _read_history(self) -> Dict:
        if not os.path.exists(self.history_file):
            return {}
        try:
            with open(self.history_file, "r") as f:
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
        except Exception:
            return {}

    def _write_history(self, history: Dict) -> None:
        dir_name = os.path.dirname(self.history_file)
        os.makedirs(dir_name, exist_ok=True)
        
        # Use same atomic write pattern as ConfigUpdater
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
            json.dump(history, tf, indent=2)
            temp_name = tf.name
        
        try:
            os.replace(temp_name, self.history_file)
        except OSError:
            # Fall back to copy and delete for Docker volumes
            import shutil
            try:
                if os.path.exists(self.history_file):
                    backup_path = f"{self.history_file}.backup"
                    shutil.copy2(self.history_file, backup_path)
                
                shutil.copy2(temp_name, self.history_file)
                
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                    
                if os.path.exists(backup_path):
                    try:
                        os.unlink(backup_path)
                    except OSError:
                        pass
            except Exception:
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, self.history_file)
                raise

    def add_download_record(self, ontology_name: str, filename: str, 
                          size_bytes: int, timestamp: str = None) -> None:
        """Add a download record for an ontology."""
        history = self._read_history()
        
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"
        
        record = {
            "filename": filename,
            "timestamp": timestamp,
            "size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes else 0
        }
        
        # Initialize list if not exists
        if ontology_name not in history:
            history[ontology_name] = []
        
        # Add new record
        history[ontology_name].append(record)
        
        # Keep only last N records
        history[ontology_name] = history[ontology_name][-self.max_records_per_ontology:]
        
        self._write_history(history)

    def get_download_history(self) -> Dict:
        """Get the full download history."""
        return self._read_history()

    def clear_history(self, ontology_name: str = None) -> None:
        """Clear download history for a specific ontology or all."""
        history = self._read_history()
        
        if ontology_name:
            if ontology_name in history:
                del history[ontology_name]
        else:
            history = {}
        
        self._write_history(history)
    
    def verify_file_exists(self, filename: str) -> bool:
        """Check if a downloaded file still exists on disk."""
        data_dir = os.environ.get("ONTOLOGY_DATA_DIR", "/app/data")
        file_path = os.path.join(data_dir, filename)
        return os.path.exists(file_path)
    
    def update_file_status(self, ontology_name: str, filename: str, status: str) -> None:
        """Update the status of a specific download record."""
        history = self._read_history()
        
        if ontology_name in history:
            for record in history[ontology_name]:
                if record.get("filename") == filename:
                    record["status"] = status
                    record["last_verified"] = datetime.utcnow().isoformat() + "Z"
                    break
        
        self._write_history(history)
    
    def verify_all_downloads(self) -> Dict[str, List[Dict]]:
        """Verify all download records and update their status."""
        history = self._read_history()
        verification_results = {}
        
        for ontology_name, records in history.items():
            verification_results[ontology_name] = []
            
            for record in records:
                filename = record.get("filename", "")
                exists = self.verify_file_exists(filename)
                
                # Update status based on file existence
                if exists:
                    record["status"] = "available"
                else:
                    record["status"] = "file_missing"
                
                record["last_verified"] = datetime.utcnow().isoformat() + "Z"
                verification_results[ontology_name].append({
                    "filename": filename,
                    "exists": exists,
                    "status": record["status"]
                })
        
        self._write_history(history)
        return verification_results
    
    def get_latest_available_download(self, ontology_name: str) -> Optional[Dict]:
        """Get the most recent download record with an available file."""
        history = self._read_history()
        
        if ontology_name not in history:
            return None
        
        # Check records in reverse order (most recent first)
        for record in reversed(history[ontology_name]):
            if self.verify_file_exists(record.get("filename", "")):
                record["status"] = "available"
                return record
        
        return None
