"""
Zenodo API Integration
Handles automated DOI minting for reproductive experimental artifacts, 
securing structural publication traceability for IEEE evaluations.
"""

import os
import requests  # type: ignore[import-untyped]
from pathlib import Path
from typing import Dict, List, Optional, Union

class ZenodoDOIClient:
    """
    Automated DOI minting pipeline integrating with the Zenodo REST Data API.
    Facilitates transparent open-source reproduction tracking for Project ASTRA.
    """
    
    ZENODO_API_URL = "https://zenodo.org/api/deposit/depositions"
    # Failsafe default pointing to the sandbox environment preventing irreversible DOI spam during testing
    ZENODO_SANDBOX_URL = "https://sandbox.zenodo.org/api/deposit/depositions"
    
    def __init__(self, access_token: Optional[str] = None, sandbox: bool = True):
        self.access_token = access_token or os.environ.get("ZENODO_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError("Zenodo Access Token must be provided or set as ZENODO_ACCESS_TOKEN env variable.")
            
        self.base_url = self.ZENODO_SANDBOX_URL if sandbox else self.ZENODO_API_URL
        self.headers = {"Content-Type": "application/json"}
        self.params = {"access_token": self.access_token}
        
    def create_deposition(self, title: str, description: str, creators: List[Dict[str, str]]) -> Dict:
        """
        Creates a new empty deposition envelope preparing for dataset upload and DOI assignment.
        """
        data = {
            "metadata": {
                "title": title,
                "upload_type": "dataset",
                "description": description,
                "creators": creators,
                "access_right": "open",
                "license": "cc-by"
            }
        }
        
        response = requests.post(
            self.base_url,
            params=self.params,
            json=data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
        
    def upload_artifact(self, deposition_id: int, file_path: Union[str, Path]) -> Dict:
        """
        Uploads an experimental configuration or parameter artifact trace to the bound deposition id.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot upload {path}, file trace does not exist locally.")
            
        # V1 File API URL configuration
        bucket_url = f"{self.base_url}/{deposition_id}/files"
        
        with open(path, "rb") as f:
            data = {"name": path.name}
            files = {"file": f}
            response = requests.post(
                bucket_url,
                params=self.params,
                data=data,
                files=files
            )
        response.raise_for_status()
        return response.json()

    def publish_and_mint_doi(self, deposition_id: int) -> str:
        """
        Finalizes the deposition via API publication layer, permanently minting the DOI.
        WARNING: This action is mathematically irreversible on the standard Zenodo API!
        """
        publish_url = f"{self.base_url}/{deposition_id}/actions/publish"
        response = requests.post(publish_url, params=self.params)
        response.raise_for_status()
        
        return response.json().get('doi', 'DOI_MINTING_FAILED')
