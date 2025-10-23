# LeafGuard - Digital Twin Integration

LeafGuard is an AI-powered leaf disease detection system with Azure Digital Twins integration for plant health monitoring and 3D visualization.

## Features

- 🔍 **AI Disease Detection**: Advanced machine learning model for tomato leaf disease identification
- 🌐 **Digital Twin Integration**: Azure Digital Twins for plant lifecycle tracking
- 📊 **Real-time Monitoring**: Live plant health status dashboard
- 🎯 **3D Visualization**: Interactive plant grid with status indicators
- 📱 **Mobile-First**: Responsive design with camera capture
- 📈 **Historical Analysis**: Track disease progression over time
- 💡 **Smart Recommendations**: Treatment suggestions based on detected diseases

## Architecture

### Frontend (React + TypeScript + Vite)
- Modern UI with shadcn/ui components
- Camera capture with file upload
- Real-time 3D plant status visualization
- Plant management and history tracking

### Backend (FastAPI + Python)
- TensorFlow-based disease detection model
- Azure Digital Twins integration
- Azure Blob Storage for image storage
- RESTful API with comprehensive endpoints

### Azure Services
- **Azure Digital Twins**: Plant twin modeling and telemetry
- **Azure Blob Storage**: Image storage with SAS token access
- **Azure Identity**: Secure authentication with DefaultAzureCredential

## Getting Started

### Prerequisites

- Python 3.8-3.11
- Node.js 16+ and npm
- Azure account with subscription
- Azure CLI installed
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd leaf-wellness-app
git checkout digital-twin-integration
```

### 2. Azure Setup (One-time)

Run the automated Azure setup script:

```powershell
# Navigate to scripts directory
cd scripts

# Run setup script (requires Azure CLI login)
.\setup-azure-resources.ps1
```

This script will:
- Create Azure resource group
- Set up Azure Digital Twins instance
- Create storage account and containers
- Create service principal for authentication
- Upload DTDL model to Azure Digital Twins
- Generate `.env` file with credentials

### 3. Backend Setup

```powershell
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify .env file exists (created by setup script)
# If not, copy from .env.template and fill in values

# Start the backend server
uvicorn main:app --host 127.0.0.1 --port 5000 --reload
```

### 4. Frontend Setup

```powershell
# Navigate to project root
cd ..

# Install dependencies
npm install

# Start development server
npm run dev
```

### 5. Create Test Plants (Optional)

```powershell
# In a new terminal, ensure backend is running
cd scripts
.\create-test-plants.ps1
```

## Usage

### Basic Workflow

1. **Access the Application**: Open `http://localhost:8080`
2. **Plant Selection**: Enter or select a plant ID (e.g., `plant_A1`)
3. **Image Capture**: Take a photo or upload an image of a tomato leaf
4. **Analysis**: Get instant disease detection results
5. **Digital Twin**: View updated plant status in 3D visualization
6. **History**: Track plant health over time

### API Endpoints

#### Core Prediction
- `POST /predict-upload` - Upload image with plant ID for analysis
- `POST /predict` - Legacy base64 image prediction

#### Plant Management
- `GET /plants` - Get all plant twins
- `GET /plant/{plant_id}` - Get specific plant details
- `POST /plant/{plant_id}/create` - Create new plant twin
- `GET /plant/{plant_id}/history` - Get plant scan history

#### 3D Visualization
- `GET /plants/status-summary` - Get plant status for 3D grid
- `POST /plant/{plant_id}/treatment` - Add treatment record

#### Health Check
- `GET /health` - Service health and ADT connection status

### Disease Classes Supported

- Tomato___Bacterial_spot
- Tomato___Early_blight
- Tomato___healthy
- Tomato___Late_blight
- Tomato___Leaf_Mold
- Tomato___Septoria_leaf_spot
- Tomato___Spider_mites_Two-spotted_spider_mite
- Tomato___Target_Spot
- Tomato___Tomato_mosaic_virus
- Tomato___Tomato_Yellow_Leaf_Curl_Virus

## Digital Twin Model

### Plant Twin Properties
- `plantId`: Unique plant identifier
- `position`: Greenhouse location (row, column, greenhouse)
- `currentHealth`: Latest health status
- `diseaseDetected`: Most recent disease detection
- `confidence`: Detection confidence score
- `visualStatus`: Status for 3D visualization (healthy/warning/critical/treatment/unknown)
- `scanHistory`: Array of historical scan results
- `treatmentHistory`: Array of applied treatments

### Telemetry
- `leafScan`: Real-time scan results
- `environmentalData`: Sensor data (future expansion)

## 3D Visualization

The 3D plant viewer provides:
- **Grid Layout**: Visual representation of greenhouse layout
- **Color Coding**: Plant health status indicators
- **Interactive Selection**: Click plants for detailed information
- **Real-time Updates**: Automatic refresh of plant status
- **Status Legend**: Clear visual indicators

### Status Colors
- 🟢 **Green**: Healthy plants
- 🟡 **Yellow**: Warning/early disease detection
- 🔴 **Red**: Critical/severe disease
- 🔵 **Blue**: Under treatment
- ⚫ **Gray**: Unknown/no recent data

## Configuration

### Environment Variables (.env)

```bash
# Azure Digital Twins
ADT_URL=https://your-adt-instance.api.wcus.digitaltwins.azure.net
STORAGE_ACCOUNT_NAME=your-storage-account
STORAGE_CONTAINER_NAME=plantimages
STORAGE_ACCOUNT_KEY=your-storage-account-key

# Azure Authentication
AZURE_CLIENT_ID=your-service-principal-client-id
AZURE_CLIENT_SECRET=your-service-principal-secret
AZURE_TENANT_ID=your-tenant-id

# Model Configuration
MODEL_PATH=plant_disease_model.h5
```

## Development

### Project Structure

```
leaf-wellness-app/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── adt_client.py        # Azure Digital Twins client
│   ├── models/
│   │   └── TomatoPlant.json # DTDL model definition
│   ├── plant_disease_model.h5
│   └── requirements.txt
├── src/
│   ├── components/
│   │   ├── LeafAnalysis.tsx     # Disease analysis component
│   │   ├── PlantViewer3D.tsx    # 3D visualization
│   │   └── CameraCapture.tsx    # Image capture
│   ├── config/
│   │   └── api.ts               # API configuration
│   └── pages/
│       └── Index.tsx            # Main application page
├── scripts/
│   ├── setup-azure-resources.ps1
│   └── create-test-plants.ps1
└── README.md
```

### Adding New Features

1. **New Disease Classes**: Update `DISEASE_MAP` in `main.py`
2. **Additional Sensors**: Extend DTDL model and telemetry endpoints
3. **New Visualizations**: Add components to the 3D viewer
4. **Custom Treatments**: Extend treatment tracking system

## Troubleshooting

### Common Issues

**Backend won't start**
- Check Python version (3.8-3.11 required)
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check `.env` file exists and has correct Azure credentials

**Digital Twin connection failed**
- Verify Azure credentials in `.env`
- Check Azure Digital Twins instance is running
- Ensure service principal has correct permissions

**Model not found**
- Place `plant_disease_model.h5` in `backend/` directory
- Check `MODEL_PATH` in `.env` file

**Frontend API errors**
- Ensure backend is running on port 5000
- Check CORS settings in `main.py`
- Verify API endpoints in `src/config/api.ts`

### Development Tips

- Use Azure Digital Twins Explorer to inspect twins and models
- Monitor Azure costs in the Azure Portal
- Check browser console for detailed error messages
- Use FastAPI's `/docs` endpoint for API documentation

## Security Considerations

- Never commit `.env` files to version control
- Use Managed Identity in production Azure environments
- Rotate service principal secrets regularly
- Implement proper authentication for production deployments
- Use HTTPS in production

## Future Enhancements

- Real IoT sensor integration (temperature, humidity, soil moisture)
- Advanced 3D models with disease visualization
- Machine learning model retraining pipeline
- Mobile app development
- Multi-plant type support (beyond tomatoes)
- Automated treatment recommendations
- Integration with irrigation systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and demonstration purposes. Please review Azure service costs and usage limits before production deployment.