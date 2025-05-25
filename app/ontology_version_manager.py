"""
Ontology Version Manager - Handles version detection, comparison, and persistent storage.
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

try:
    from .config import ONTOLOGY_CONFIG
except ImportError:
    # For standalone testing
    ONTOLOGY_CONFIG = {}


class OntologyVersionManager:
    """Manages ontology versions to avoid re-embedding identical data."""
    
    def __init__(self, data_dir: str = None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir or os.getenv("ONTOLOGY_DATA_DIR", "./data/ontologies")
        self.ensure_data_directory()
    
    def ensure_data_directory(self) -> None:
        """Ensure the data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
        self.logger.info(f"Ontology data directory: {self.data_dir}")
    
    def extract_go_version_info(self, go_data: Dict) -> Dict[str, str]:
        """Extract version information from GO.json data."""
        version_info = {
            "version_date": "",
            "version_url": "",
            "format_version": "",
            "graph_id": ""
        }
        
        try:
            graphs = go_data.get("graphs", [])
            if not graphs:
                return version_info
            
            graph = graphs[0]
            version_info["graph_id"] = graph.get("id", "")
            
            meta = graph.get("meta", {})
            version_info["version_url"] = meta.get("version", "")
            
            # Extract from basicPropertyValues
            basic_props = meta.get("basicPropertyValues", [])
            for prop in basic_props:
                pred = prop.get("pred", "")
                val = prop.get("val", "")
                
                if pred == "http://www.w3.org/2002/07/owl#versionInfo":
                    version_info["version_date"] = val
                elif pred == "http://www.geneontology.org/formats/oboInOwl#hasOBOFormatVersion":
                    version_info["format_version"] = val
            
        except Exception as e:
            self.logger.warning(f"Failed to extract GO version info: {e}")
        
        return version_info
    
    def generate_version_hash(self, ontology_data: Dict) -> str:
        """Generate a hash of the ontology data for version comparison."""
        # Create a stable hash based on version info and content size
        version_info = self.extract_go_version_info(ontology_data)
        
        # Use version date and graph count for hashing
        graphs = ontology_data.get("graphs", [])
        nodes_count = len(graphs[0].get("nodes", [])) if graphs else 0
        
        hash_data = {
            "version_date": version_info["version_date"],
            "version_url": version_info["version_url"],
            "nodes_count": nodes_count,
            "graph_id": version_info["graph_id"]
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    def get_version_metadata_path(self, ontology_name: str) -> str:
        """Get path for version metadata file."""
        return os.path.join(self.data_dir, f"{ontology_name}_version_metadata.json")
    
    def get_stored_version_info(self, ontology_name: str) -> Optional[Dict]:
        """Get stored version information for an ontology."""
        metadata_path = self.get_version_metadata_path(ontology_name)
        
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to read version metadata: {e}")
            return None
    
    def store_version_info(
        self, 
        ontology_name: str, 
        version_info: Dict, 
        collection_name: str,
        source_url: str
    ) -> None:
        """Store version information after successful embedding."""
        metadata = {
            "ontology_name": ontology_name,
            "version_info": version_info,
            "collection_name": collection_name,
            "source_url": source_url,
            "stored_at": datetime.now().isoformat(),
            "data_directory": self.data_dir
        }
        
        metadata_path = self.get_version_metadata_path(ontology_name)
        
        try:
            # Use atomic write with fallback for Docker environments
            import tempfile
            dir_name = os.path.dirname(metadata_path)
            os.makedirs(dir_name, exist_ok=True)
            
            with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False) as tf:
                json.dump(metadata, tf, indent=2)
                temp_name = tf.name
            
            # Try atomic replace first
            try:
                os.replace(temp_name, metadata_path)
            except OSError:
                # Fall back to copy method for Docker volumes
                import shutil
                shutil.copy2(temp_name, metadata_path)
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
            
            self.logger.info(f"Stored version metadata for {ontology_name} at {metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed to store version metadata: {e}")
    
    def compare_versions(
        self, 
        ontology_name: str, 
        new_ontology_data: Dict
    ) -> Tuple[bool, Optional[Dict], Dict]:
        """
        Compare new ontology data with stored version.
        
        Returns:
            - needs_update: bool (True if update needed)
            - stored_info: Dict or None (existing version info)
            - new_version_info: Dict (new version info)
        """
        new_version_info = self.extract_go_version_info(new_ontology_data)
        new_hash = self.generate_version_hash(new_ontology_data)
        new_version_info["content_hash"] = new_hash
        
        stored_info = self.get_stored_version_info(ontology_name)
        
        if not stored_info:
            self.logger.info(f"No stored version found for {ontology_name}")
            return True, None, new_version_info
        
        stored_version_info = stored_info.get("version_info", {})
        stored_hash = stored_version_info.get("content_hash", "")
        
        if new_hash != stored_hash:
            self.logger.info(
                f"Version change detected for {ontology_name}: "
                f"{stored_version_info.get('version_date')} -> {new_version_info['version_date']}"
            )
            return True, stored_info, new_version_info
        else:
            self.logger.info(f"No version change for {ontology_name}, using cached data")
            return False, stored_info, new_version_info
    
    def get_weaviate_data_path(self) -> str:
        """Get path where Weaviate should store its persistent data."""
        weaviate_data_path = os.path.join(self.data_dir, "weaviate_data")
        os.makedirs(weaviate_data_path, exist_ok=True)
        return weaviate_data_path
    
    def cleanup_old_collections(self, current_collections: list) -> None:
        """Clean up metadata for collections that no longer exist."""
        if not os.path.exists(self.data_dir):
            return
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith("_version_metadata.json"):
                metadata_path = os.path.join(self.data_dir, filename)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    collection_name = metadata.get("collection_name", "")
                    if collection_name and collection_name not in current_collections:
                        os.remove(metadata_path)
                        self.logger.info(f"Cleaned up metadata for removed collection: {collection_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to process metadata file {filename}: {e}")
    
    def list_stored_versions(self) -> Dict[str, Dict]:
        """List all stored ontology versions."""
        versions = {}
        
        if not os.path.exists(self.data_dir):
            return versions
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith("_version_metadata.json"):
                ontology_name = filename.replace("_version_metadata.json", "")
                metadata = self.get_stored_version_info(ontology_name)
                if metadata:
                    versions[ontology_name] = metadata
        
        return versions


def extract_version_from_ontology_data(ontology_data: Dict, ontology_format: str = "json") -> Dict[str, str]:
    """
    Extract version information from different ontology formats.
    Currently supports GO JSON format, can be extended for other formats.
    """
    if ontology_format.lower() == "json":
        version_manager = OntologyVersionManager()
        return version_manager.extract_go_version_info(ontology_data)
    else:
        # Placeholder for other formats (OBO, OWL, etc.)
        return {
            "version_date": "",
            "version_url": "",
            "format_version": "",
            "graph_id": ""
        }


# Example usage and testing
if __name__ == "__main__":
    # Test with real GO data
    import json
    
    print("Testing OntologyVersionManager...")
    
    version_manager = OntologyVersionManager("./test_data")
    
    # Load GO data
    with open("go.json", "r") as f:
        go_data = json.load(f)
    
    # Extract version info
    version_info = version_manager.extract_go_version_info(go_data)
    print(f"Version info: {version_info}")
    
    # Generate hash
    content_hash = version_manager.generate_version_hash(go_data)
    print(f"Content hash: {content_hash}")
    
    # Test version comparison
    needs_update, stored, new_info = version_manager.compare_versions("GO", go_data)
    print(f"Needs update: {needs_update}")
    print(f"New version: {new_info['version_date']}")