# Azure Digital Twins Setup Script
# Run this script to create Azure resources for LeafGuard Digital Twin integration

# Variables - Update these with your preferred names
$resourceGroup = "rg-leafguard"
$location = "eastus"
$adtInstance = "leafguard-dt-$(Get-Random -Minimum 1000 -Maximum 9999)"
$storageAccount = "leafguardstorage$(Get-Random -Minimum 1000 -Maximum 9999)"
$containerName = "plantimages"
$appRegistrationName = "leafguard-app"

Write-Host "Setting up Azure resources for LeafGuard Digital Twin integration..." -ForegroundColor Green

# 1. Login to Azure (if not already logged in)
Write-Host "Checking Azure login status..." -ForegroundColor Yellow
try {
    $account = az account show --output table 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Please login to Azure..." -ForegroundColor Yellow
        az login
    } else {
        Write-Host "Already logged in to Azure" -ForegroundColor Green
    }
} catch {
    Write-Host "Please login to Azure..." -ForegroundColor Yellow
    az login
}

# 2. Create Resource Group
Write-Host "Creating resource group: $resourceGroup" -ForegroundColor Yellow
az group create --name $resourceGroup --location $location

# 3. Create Azure Digital Twins instance
Write-Host "Creating Azure Digital Twins instance: $adtInstance" -ForegroundColor Yellow
az dt create --dt-name $adtInstance --resource-group $resourceGroup --location $location

# 4. Create Storage Account
Write-Host "Creating storage account: $storageAccount" -ForegroundColor Yellow
az storage account create `
    --name $storageAccount `
    --resource-group $resourceGroup `
    --location $location `
    --sku Standard_LRS `
    --kind StorageV2

# 5. Create storage container
Write-Host "Creating storage container: $containerName" -ForegroundColor Yellow
az storage container create `
    --account-name $storageAccount `
    --name $containerName `
    --auth-mode login

# 6. Create App Registration for authentication
Write-Host "Creating app registration: $appRegistrationName" -ForegroundColor Yellow
$appId = az ad app create --display-name $appRegistrationName --query appId --output tsv

# 7. Create service principal
Write-Host "Creating service principal..." -ForegroundColor Yellow
$spOutput = az ad sp create-for-rbac --name $appId --role "Azure Digital Twins Data Owner" --scopes "/subscriptions/$(az account show --query id --output tsv)/resourceGroups/$resourceGroup" --query "{clientId:appId, clientSecret:password, tenantId:tenant}" --output json
$spData = $spOutput | ConvertFrom-Json

# 8. Get storage account key
Write-Host "Retrieving storage account key..." -ForegroundColor Yellow
$storageKey = az storage account keys list --resource-group $resourceGroup --account-name $storageAccount --query "[0].value" --output tsv

# 9. Get ADT endpoint
$adtEndpoint = az dt show --dt-name $adtInstance --resource-group $resourceGroup --query hostName --output tsv

# 10. Upload DTDL model
Write-Host "Uploading DTDL model to Azure Digital Twins..." -ForegroundColor Yellow
$modelPath = "..\backend\models\TomatoPlant.json"
if (Test-Path $modelPath) {
    az dt model create --dt-name $adtInstance --models $modelPath
    Write-Host "DTDL model uploaded successfully" -ForegroundColor Green
} else {
    Write-Host "Warning: DTDL model file not found at $modelPath" -ForegroundColor Red
}

# 11. Generate .env file
Write-Host "Generating .env file..." -ForegroundColor Yellow
$envContent = @"
# Azure Digital Twins Configuration
ADT_URL=https://$adtEndpoint
STORAGE_ACCOUNT_NAME=$storageAccount
STORAGE_CONTAINER_NAME=$containerName
STORAGE_ACCOUNT_KEY=$storageKey

# Azure Authentication
AZURE_CLIENT_ID=$($spData.clientId)
AZURE_CLIENT_SECRET=$($spData.clientSecret)
AZURE_TENANT_ID=$($spData.tenantId)

# Model Configuration
MODEL_PATH=plant_disease_model.h5
"@

$envPath = "..\backend\.env"
$envContent | Out-File -FilePath $envPath -Encoding UTF8

Write-Host "`n" -NoNewline
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Azure resources created successfully!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "`n" -NoNewline

Write-Host "Resource Details:" -ForegroundColor Cyan
Write-Host "- Resource Group: $resourceGroup" -ForegroundColor White
Write-Host "- Digital Twins Instance: $adtInstance" -ForegroundColor White
Write-Host "- Storage Account: $storageAccount" -ForegroundColor White
Write-Host "- ADT Endpoint: https://$adtEndpoint" -ForegroundColor White

Write-Host "`n" -NoNewline
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Review the generated .env file at: $envPath" -ForegroundColor White
Write-Host "2. Install Python dependencies: pip install -r requirements.txt" -ForegroundColor White
Write-Host "3. Test the backend: uvicorn main:app --host 127.0.0.1 --port 5000 --reload" -ForegroundColor White
Write-Host "4. Check the /health endpoint to verify ADT connection" -ForegroundColor White

Write-Host "`n" -NoNewline
Write-Host "Important:" -ForegroundColor Red
Write-Host "- The .env file contains sensitive information" -ForegroundColor White
Write-Host "- Do not commit the .env file to version control" -ForegroundColor White
Write-Host "- Store credentials securely in production" -ForegroundColor White