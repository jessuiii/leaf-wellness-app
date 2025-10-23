# Sample script to create test plants in Azure Digital Twins
# Run this after setting up Azure resources

# Test plant data for 3D greenhouse simulation
$testPlants = @(
    @{
        id = "plant_A1"
        position = @{ row = "A"; column = 1; greenhouse = "Greenhouse_1" }
        location = "Greenhouse 1, Row A, Position 1"
    },
    @{
        id = "plant_A2" 
        position = @{ row = "A"; column = 2; greenhouse = "Greenhouse_1" }
        location = "Greenhouse 1, Row A, Position 2"
    },
    @{
        id = "plant_A3"
        position = @{ row = "A"; column = 3; greenhouse = "Greenhouse_1" }
        location = "Greenhouse 1, Row A, Position 3"
    },
    @{
        id = "plant_B1"
        position = @{ row = "B"; column = 1; greenhouse = "Greenhouse_1" }
        location = "Greenhouse 1, Row B, Position 1"
    },
    @{
        id = "plant_B2"
        position = @{ row = "B"; column = 2; greenhouse = "Greenhouse_1" }
        location = "Greenhouse 1, Row B, Position 2"
    },
    @{
        id = "plant_B3"
        position = @{ row = "B"; column = 3; greenhouse = "Greenhouse_1" }
        location = "Greenhouse 1, Row B, Position 3"
    },
    @{
        id = "plant_C1"
        position = @{ row = "C"; column = 1; greenhouse = "Greenhouse_1" }
        location = "Greenhouse 1, Row C, Position 1"
    },
    @{
        id = "plant_C2"
        position = @{ row = "C"; column = 2; greenhouse = "Greenhouse_1" }
        location = "Greenhouse 1, Row C, Position 2"
    },
    @{
        id = "plant_C3"
        position = @{ row = "C"; column = 3; greenhouse = "Greenhouse_1" }
        location = "Greenhouse 1, Row C, Position 3"
    }
)

Write-Host "Creating test plants in Azure Digital Twins..." -ForegroundColor Green

# Assuming backend server is running on localhost:5000
$apiBase = "http://127.0.0.1:5000"

foreach ($plant in $testPlants) {
    Write-Host "Creating plant: $($plant.id)" -ForegroundColor Yellow
    
    $body = @{
        plant_id = $plant.id
        position = $plant.position
        location = $plant.location
    } | ConvertTo-Json -Depth 3
    
    try {
        $response = Invoke-RestMethod -Uri "$apiBase/plant/$($plant.id)/create" -Method Post -Body $body -ContentType "application/json"
        Write-Host "✓ Successfully created $($plant.id)" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to create $($plant.id): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`nTest plants creation completed!" -ForegroundColor Green
Write-Host "You can now view plants at: $apiBase/plants" -ForegroundColor Cyan