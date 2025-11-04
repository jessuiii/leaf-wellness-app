#!/usr/bin/env python3
"""
Real FastAPI server with your trained DenseNet model integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import os
import tempfile
import json
import requests
from azure_config import get_glb_url, get_storage_info
from smart_glb_assigner import SmartGLBAssigner
# Removed database_api import to avoid Cosmos DB dependency

# Initialize FastAPI app
app = FastAPI(title="Leaf Guard 3D API - DenseNet Integration", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database functionality is handled by azure_table_api.py on port 8001

# Initialize the smart GLB assigner with your trained model
try:
    smart_assigner = SmartGLBAssigner()
    print("✅ DenseNet model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading DenseNet model: {e}")
    smart_assigner = None

@app.get("/")
async def root():
    """Root endpoint"""
    model_status = "loaded" if smart_assigner else "failed_to_load"
    return {
        "message": "Leaf Guard 3D API - DenseNet Integration",
        "status": "running",
        "model_status": model_status,
        "endpoints": [
            "/analyze-leaf-real",
            "/glb-urls", 
            "/storage-info",
            "/model-info",
            "/api/plants",
            "/api/health-stats",
            "/api/database-status"
        ]
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded DenseNet model"""
    if not smart_assigner:
        return {
            "success": False,
            "error": "DenseNet model not loaded"
        }
    
    try:
        model_info = {
            "model_loaded": True,
            "model_path": smart_assigner.model_path,
            "device": str(smart_assigner.densenet_model.device),
            "num_classes": smart_assigner.densenet_model.num_classes,
            "class_names": smart_assigner.densenet_model.class_names,
            "glb_files": smart_assigner.glb_files
        }
        return {
            "success": True,
            "model_info": model_info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/storage-info")
async def get_storage_information():
    """Get Azure storage information"""
    try:
        storage_info = get_storage_info()
        return {
            "success": True,
            "storage_info": storage_info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/glb-urls")
async def get_glb_urls():
    """Get GLB file URLs (proxied to avoid CORS)"""
    try:
        return {
            "success": True,
            "glb_urls": {
                "healthy": convert_azure_url_to_proxy(get_glb_url("healthy")),
                "diseased": convert_azure_url_to_proxy(get_glb_url("diseased"))
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/analyze-leaf")
async def analyze_leaf_image(file: UploadFile = File(...)):
    """
    Analyze leaf image using your trained DenseNet model
    """
    if not smart_assigner:
        raise HTTPException(status_code=500, detail="DenseNet model not loaded")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        print(f"🔍 Analyzing image: {file.filename} ({len(content)} bytes)")
        
        # Use your trained DenseNet model for analysis
        analysis_result = smart_assigner.densenet_model.predict_image(temp_file_path)
        
        print(f"📊 Raw analysis result: {analysis_result}")
        print(f"📊 Analysis result type: {type(analysis_result)}")
        print(f"📊 Analysis result keys: {analysis_result.keys() if isinstance(analysis_result, dict) else 'Not a dict'}")
        
        # Get the appropriate GLB URL based on health status
        if 'error' not in analysis_result:
            glb_url = smart_assigner.glb_files.get(analysis_result['health_status'], 
                                                  smart_assigner.glb_files['healthy'])
            # Convert to proxy URL to avoid CORS issues
            glb_url = convert_azure_url_to_proxy(glb_url)
            
            response = {
                "filename": file.filename,
                "file_size": len(content),
                "health_status": analysis_result['health_status'],
                "disease_type": analysis_result['disease_type'],
                "predicted_class": analysis_result['predicted_class'],
                "confidence": analysis_result['confidence'],
                "class_index": analysis_result['class_index'],
                "glb_recommendation": analysis_result['glb_recommendation'],
                "glb_url": glb_url,
                "analysis_method": "densenet_trained_model"
            }
        else:
            # Convert fallback URL to proxy as well
            fallback_glb = convert_azure_url_to_proxy(smart_assigner.glb_files['healthy'])
            response = {
                "filename": file.filename,
                "file_size": len(content),
                "error": analysis_result['error'],
                "health_status": "unknown",
                "glb_url": fallback_glb
            }

        # Clean up temp file
        os.unlink(temp_file_path)

        return {
            "success": True,
            "analysis": response
        }

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
        print(f"❌ Error during analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/proxy-glb/{filename}")
@app.head("/proxy-glb/{filename}")
async def proxy_glb(filename: str):
    """
    Proxy endpoint to serve GLB files from Azure storage without CORS issues
    """
    try:
        # Get the Azure URL for the GLB file
        azure_url = f"https://leafguardstorage.blob.core.windows.net/glb-models/{filename}"
        
        # Fetch the file from Azure
        response = requests.get(azure_url, stream=True)
        
        if response.status_code == 200:
            # Stream the file content with proper headers
            return StreamingResponse(
                iter([response.content]),
                media_type="model/gltf-binary",
                headers={
                    "Content-Disposition": f"inline; filename={filename}",
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "public, max-age=3600"
                }
            )
        else:
            raise HTTPException(status_code=404, detail=f"GLB file not found: {filename}")
            
    except requests.RequestException as e:
        print(f"❌ Error fetching GLB from Azure: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching GLB file: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error in GLB proxy: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def convert_azure_url_to_proxy(azure_url: str) -> str:
    """
    Convert Azure storage URL to local proxy URL to avoid CORS issues
    """
    if "leafguardstorage.blob.core.windows.net/glb-models/" in azure_url:
        filename = azure_url.split("/")[-1]
        return f"http://localhost:8000/proxy-glb/{filename}"
    return azure_url

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Leaf Guard Main API...")
    print("📊 DenseNet Model: Loaded")
    print("🌐 API will be available at: http://localhost:8002")
    print("📖 API docs at: http://localhost:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8002)