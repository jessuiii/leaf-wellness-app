import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from azure.digitaltwins.core import DigitalTwinsClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ADTClient:
    """Azure Digital Twins client for managing plant twins and telemetry."""
    
    def __init__(self):
        """Initialize the ADT client with Azure credentials."""
        self.adt_url = os.getenv("ADT_URL", "https://leafguard-dt.api.wcus.digitaltwins.azure.net")
        self.storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME", "leafguardstorage")
        self.storage_container_name = os.getenv("STORAGE_CONTAINER_NAME", "plantimages")
        
        try:
            # Initialize Azure credentials
            credential = DefaultAzureCredential()
            
            # Initialize ADT client
            self.adt_client = DigitalTwinsClient(self.adt_url, credential)
            
            # Initialize Blob Storage client
            storage_url = f"https://{self.storage_account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(storage_url, credential)
            
            logger.info("ADT Client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ADT Client: {str(e)}")
            raise
    
    def create_plant_twin(self, plant_id: str, position: Dict[str, Any], location: str = "Unknown") -> bool:
        """
        Create a new plant twin in Azure Digital Twins.
        
        Args:
            plant_id: Unique identifier for the plant
            position: Dictionary with row, column, greenhouse info
            location: Human-readable location description
            
        Returns:
            bool: True if successful, False otherwise
        """
        twin_data = {
            "$metadata": {"$model": "dtmi:leafguard:TomatoPlant;1"},
            "plantId": plant_id,
            "position": position,
            "currentHealth": "Unknown",
            "diseaseDetected": "None",
            "confidence": 0.0,
            "visualStatus": "unknown",
            "scanHistory": [],
            "treatmentHistory": []
        }
        
        try:
            self.adt_client.upsert_digital_twin(plant_id, json.dumps(twin_data))
            logger.info(f"Successfully created/updated twin for plant {plant_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating twin for plant {plant_id}: {str(e)}")
            return False
    
    def update_plant_scan(self, plant_id: str, disease: str, confidence: float, image_url: str) -> bool:
        """
        Update plant twin with new scan results.
        
        Args:
            plant_id: Plant identifier
            disease: Detected disease name
            confidence: Detection confidence score
            image_url: URL to the uploaded image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current twin data
            twin_json = self.adt_client.get_digital_twin(plant_id)
            twin_data = json.loads(twin_json)
            
            # Prepare new scan entry
            new_scan = {
                "timestamp": datetime.now().isoformat(),
                "disease": disease,
                "confidence": confidence,
                "imageUrl": image_url
            }
            
            # Update scan history
            if "scanHistory" not in twin_data:
                twin_data["scanHistory"] = []
            
            twin_data["scanHistory"].append(new_scan)
            
            # Keep only last 50 scans to avoid data bloat
            if len(twin_data["scanHistory"]) > 50:
                twin_data["scanHistory"] = twin_data["scanHistory"][-50:]
            
            # Determine visual status based on disease
            visual_status = self._determine_visual_status(disease, confidence)
            
            # Create update patch
            patch = [
                {"op": "replace", "path": "/currentHealth", "value": disease},
                {"op": "replace", "path": "/diseaseDetected", "value": disease},
                {"op": "replace", "path": "/confidence", "value": confidence},
                {"op": "replace", "path": "/lastScanDate", "value": datetime.now().isoformat()},
                {"op": "replace", "path": "/visualStatus", "value": visual_status},
                {"op": "replace", "path": "/scanHistory", "value": twin_data["scanHistory"]}
            ]
            
            # Update the twin
            self.adt_client.update_digital_twin(plant_id, patch)
            
            # Send telemetry
            telemetry_data = {
                "disease": disease,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "imageUrl": image_url
            }
            self.adt_client.publish_telemetry(plant_id, telemetry_data)
            
            logger.info(f"Successfully updated plant {plant_id} with disease: {disease}")
            return True
            
        except ResourceNotFoundError:
            # Twin doesn't exist, create it first
            logger.warning(f"Twin {plant_id} not found, creating new twin")
            position = {"row": "Unknown", "column": 0, "greenhouse": "Default"}
            if self.create_plant_twin(plant_id, position):
                return self.update_plant_scan(plant_id, disease, confidence, image_url)
            return False
            
        except Exception as e:
            logger.error(f"Error updating plant {plant_id}: {str(e)}")
            return False
    
    def get_plant_twin(self, plant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete twin data for a plant.
        
        Args:
            plant_id: Plant identifier
            
        Returns:
            Dict containing twin data or None if not found
        """
        try:
            twin_json = self.adt_client.get_digital_twin(plant_id)
            return json.loads(twin_json)
        except ResourceNotFoundError:
            logger.warning(f"Twin {plant_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error getting twin {plant_id}: {str(e)}")
            return None
    
    def get_plant_history(self, plant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get scan history for a plant.
        
        Args:
            plant_id: Plant identifier
            limit: Maximum number of history entries to return
            
        Returns:
            List of scan history entries
        """
        try:
            twin_data = self.get_plant_twin(plant_id)
            if twin_data and "scanHistory" in twin_data:
                history = twin_data["scanHistory"]
                # Return most recent entries first
                return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
            return []
        except Exception as e:
            logger.error(f"Error getting history for plant {plant_id}: {str(e)}")
            return []
    
    def get_all_plants(self) -> List[Dict[str, Any]]:
        """
        Get all plant twins.
        
        Returns:
            List of all plant twin data
        """
        try:
            query = "SELECT * FROM digitaltwins WHERE IS_OF_MODEL('dtmi:leafguard:TomatoPlant;1')"
            query_result = self.adt_client.query_twins(query)
            return [json.loads(item) for item in query_result]
        except Exception as e:
            logger.error(f"Error getting all plants: {str(e)}")
            return []
    
    def upload_image_to_blob(self, image_data: bytes, plant_id: str, filename: str = None) -> str:
        """
        Upload image to Azure Blob Storage.
        
        Args:
            image_data: Binary image data
            plant_id: Plant identifier
            filename: Optional filename, auto-generated if not provided
            
        Returns:
            str: URL to the uploaded image
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{plant_id}_{timestamp}.jpg"
            
            blob_name = f"{plant_id}/{filename}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.storage_container_name, 
                blob=blob_name
            )
            
            # Upload the image
            blob_client.upload_blob(image_data, overwrite=True)
            
            # Generate a SAS URL for access (valid for 1 year)
            sas_url = self._generate_blob_sas_url(blob_name)
            
            logger.info(f"Successfully uploaded image for plant {plant_id}")
            return sas_url
            
        except Exception as e:
            logger.error(f"Error uploading image for plant {plant_id}: {str(e)}")
            raise
    
    def add_treatment_record(self, plant_id: str, treatment: str, dosage: str = "", notes: str = "") -> bool:
        """
        Add a treatment record to a plant's history.
        
        Args:
            plant_id: Plant identifier
            treatment: Treatment applied
            dosage: Dosage information
            notes: Additional notes
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            twin_data = self.get_plant_twin(plant_id)
            if not twin_data:
                return False
            
            new_treatment = {
                "timestamp": datetime.now().isoformat(),
                "treatment": treatment,
                "dosage": dosage,
                "notes": notes
            }
            
            if "treatmentHistory" not in twin_data:
                twin_data["treatmentHistory"] = []
            
            twin_data["treatmentHistory"].append(new_treatment)
            
            # Keep only last 20 treatments
            if len(twin_data["treatmentHistory"]) > 20:
                twin_data["treatmentHistory"] = twin_data["treatmentHistory"][-20:]
            
            patch = [
                {"op": "replace", "path": "/treatmentHistory", "value": twin_data["treatmentHistory"]}
            ]
            
            self.adt_client.update_digital_twin(plant_id, patch)
            logger.info(f"Added treatment record for plant {plant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding treatment for plant {plant_id}: {str(e)}")
            return False
    
    def _determine_visual_status(self, disease: str, confidence: float) -> str:
        """
        Determine visual status based on disease and confidence.
        
        Args:
            disease: Detected disease
            confidence: Detection confidence
            
        Returns:
            str: Visual status for 3D display
        """
        if disease.lower() in ["healthy", "tomato___healthy"]:
            return "healthy"
        elif confidence < 0.5:
            return "unknown"
        elif confidence < 0.7:
            return "warning"
        else:
            return "critical"
    
    def _generate_blob_sas_url(self, blob_name: str) -> str:
        """
        Generate a SAS URL for blob access.
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            str: SAS URL
        """
        try:
            from datetime import datetime, timedelta
            
            # Generate SAS token (valid for 1 year)
            sas_token = generate_blob_sas(
                account_name=self.storage_account_name,
                container_name=self.storage_container_name,
                blob_name=blob_name,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=365),
                account_key=os.getenv("STORAGE_ACCOUNT_KEY")
            )
            
            return f"https://{self.storage_account_name}.blob.core.windows.net/{self.storage_container_name}/{blob_name}?{sas_token}"
            
        except Exception as e:
            logger.warning(f"Could not generate SAS URL, returning direct URL: {str(e)}")
            return f"https://{self.storage_account_name}.blob.core.windows.net/{self.storage_container_name}/{blob_name}"