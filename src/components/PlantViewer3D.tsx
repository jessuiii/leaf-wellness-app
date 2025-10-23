import React, { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Loader2, RotateCcw, ZoomIn, ZoomOut, Info } from 'lucide-react';
import { apiEndpoints } from '@/config/api';

interface PlantStatus {
  plant_id: string;
  position: {
    row: string;
    column: number;
    greenhouse: string;
  };
  current_health: string;
  last_scan_date?: string;
  confidence: number;
}

interface PlantStatusSummary {
  healthy: PlantStatus[];
  warning: PlantStatus[];
  critical: PlantStatus[];
  treatment: PlantStatus[];
  unknown: PlantStatus[];
}

export const PlantViewer3D: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [statusSummary, setStatusSummary] = useState<PlantStatusSummary | null>(null);
  const [selectedPlant, setSelectedPlant] = useState<PlantStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Status color mapping
  const statusColors = {
    healthy: '#22c55e',
    warning: '#eab308', 
    critical: '#ef4444',
    treatment: '#3b82f6',
    unknown: '#6b7280'
  };

  useEffect(() => {
    loadPlantStatus();
    const interval = setInterval(loadPlantStatus, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (statusSummary) {
      renderPlantGrid();
    }
  }, [statusSummary, selectedPlant]);

  const loadPlantStatus = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(apiEndpoints.statusSummary());
      
      if (!response.ok) {
        throw new Error('Digital twin service not available');
      }
      
      const data = await response.json();
      setStatusSummary(data.summary);
      setLastUpdated(new Date(data.last_updated));
      
    } catch (err: any) {
      setError(err.message);
      console.error('Failed to load plant status:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const renderPlantGrid = () => {
    const canvas = canvasRef.current;
    if (!canvas || !statusSummary) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = 800;
    canvas.height = 600;

    // Clear canvas
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    drawGrid(ctx);

    // Draw plants
    drawPlants(ctx);

    // Draw legend
    drawLegend(ctx);
  };

  const drawGrid = (ctx: CanvasRenderingContext2D) => {
    const startX = 100;
    const startY = 100;
    const cellSize = 80;
    const rows = 4;
    const cols = 4;

    // Draw grid lines
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;

    // Vertical lines
    for (let i = 0; i <= cols; i++) {
      ctx.beginPath();
      ctx.moveTo(startX + i * cellSize, startY);
      ctx.lineTo(startX + i * cellSize, startY + rows * cellSize);
      ctx.stroke();
    }

    // Horizontal lines
    for (let i = 0; i <= rows; i++) {
      ctx.beginPath();
      ctx.moveTo(startX, startY + i * cellSize);
      ctx.lineTo(startX + cols * cellSize, startY + i * cellSize);
      ctx.stroke();
    }

    // Draw row labels
    ctx.fillStyle = '#374151';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    
    const rowLabels = ['A', 'B', 'C'];
    rowLabels.forEach((label, index) => {
      ctx.fillText(label, startX - 30, startY + (index + 0.5) * cellSize + 5);
    });

    // Draw column labels
    for (let i = 1; i <= 3; i++) {
      ctx.fillText(i.toString(), startX + (i - 0.5) * cellSize, startY - 20);
    }

    // Draw title
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText('Greenhouse 1 - Plant Status Grid', 400, 40);
  };

  const drawPlants = (ctx: CanvasRenderingContext2D) => {
    if (!statusSummary) return;

    const startX = 100;
    const startY = 100;
    const cellSize = 80;

    // Combine all plants
    const allPlants = [
      ...statusSummary.healthy,
      ...statusSummary.warning,
      ...statusSummary.critical,
      ...statusSummary.treatment,
      ...statusSummary.unknown
    ];

    allPlants.forEach(plant => {
      const position = plant.position;
      
      // Convert row letter to index
      const rowIndex = position.row.charCodeAt(0) - 'A'.charCodeAt(0);
      const colIndex = position.column - 1;

      if (rowIndex >= 0 && rowIndex < 3 && colIndex >= 0 && colIndex < 3) {
        const x = startX + (colIndex + 0.5) * cellSize;
        const y = startY + (rowIndex + 0.5) * cellSize;

        // Determine plant status and color
        let status = 'unknown';
        for (const [statusType, plants] of Object.entries(statusSummary)) {
          if (plants.some((p: PlantStatus) => p.plant_id === plant.plant_id)) {
            status = statusType;
            break;
          }
        }

        // Draw plant circle
        const radius = selectedPlant?.plant_id === plant.plant_id ? 25 : 20;
        ctx.fillStyle = statusColors[status as keyof typeof statusColors];
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();

        // Draw border for selected plant
        if (selectedPlant?.plant_id === plant.plant_id) {
          ctx.strokeStyle = '#1f2937';
          ctx.lineWidth = 3;
          ctx.stroke();
        }

        // Draw plant ID
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(plant.plant_id.replace('plant_', ''), x, y + 3);

        // Store plant position for click detection
        (plant as any).canvasX = x;
        (plant as any).canvasY = y;
        (plant as any).radius = radius;
      }
    });
  };

  const drawLegend = (ctx: CanvasRenderingContext2D) => {
    const legendX = 450;
    const legendY = 150;
    const itemHeight = 30;

    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(legendX - 10, legendY - 20, 250, 200);
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX - 10, legendY - 20, 250, 200);

    // Legend title
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Plant Health Status', legendX, legendY);

    // Legend items
    const legendItems = [
      { status: 'healthy', label: 'Healthy', count: statusSummary?.healthy.length || 0 },
      { status: 'warning', label: 'Warning', count: statusSummary?.warning.length || 0 },
      { status: 'critical', label: 'Critical', count: statusSummary?.critical.length || 0 },
      { status: 'treatment', label: 'Treatment', count: statusSummary?.treatment.length || 0 },
      { status: 'unknown', label: 'Unknown', count: statusSummary?.unknown.length || 0 }
    ];

    ctx.font = '12px sans-serif';
    legendItems.forEach((item, index) => {
      const y = legendY + 30 + index * itemHeight;

      // Draw color circle
      ctx.fillStyle = statusColors[item.status as keyof typeof statusColors];
      ctx.beginPath();
      ctx.arc(legendX + 10, y - 5, 8, 0, 2 * Math.PI);
      ctx.fill();

      // Draw label and count
      ctx.fillStyle = '#374151';
      ctx.fillText(`${item.label} (${item.count})`, legendX + 25, y);
    });
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!statusSummary) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Check if click is on any plant
    const allPlants = [
      ...statusSummary.healthy,
      ...statusSummary.warning,
      ...statusSummary.critical,
      ...statusSummary.treatment,
      ...statusSummary.unknown
    ];

    for (const plant of allPlants) {
      const plantData = plant as any;
      if (plantData.canvasX && plantData.canvasY) {
        const distance = Math.sqrt(
          Math.pow(clickX - plantData.canvasX, 2) + 
          Math.pow(clickY - plantData.canvasY, 2)
        );

        if (distance <= plantData.radius) {
          setSelectedPlant(plant);
          return;
        }
      }
    }

    // Click outside any plant
    setSelectedPlant(null);
  };

  const getTotalPlants = () => {
    if (!statusSummary) return 0;
    return Object.values(statusSummary).reduce((total, plants) => total + plants.length, 0);
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'healthy': return 'default';
      case 'warning': return 'secondary';
      case 'critical': return 'destructive';
      case 'treatment': return 'outline';
      default: return 'secondary';
    }
  };

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin mr-2" />
          <span>Loading plant status...</span>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <div className="text-center">
          <div className="text-red-600 mb-4">⚠️ {error}</div>
          <Button onClick={loadPlantStatus} variant="outline">
            <RotateCcw className="mr-2 h-4 w-4" />
            Retry
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Status Summary */}
      <Card className="p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Plant Health Overview</h3>
          <div className="text-sm text-muted-foreground">
            {lastUpdated && `Last updated: ${lastUpdated.toLocaleTimeString()}`}
          </div>
        </div>
        
        <div className="flex flex-wrap gap-2 mb-4">
          {statusSummary && Object.entries(statusSummary).map(([status, plants]) => (
            <Badge 
              key={status} 
              variant={getStatusBadgeVariant(status)}
              className="capitalize"
            >
              {status}: {plants.length}
            </Badge>
          ))}
          <Badge variant="outline">
            Total: {getTotalPlants()}
          </Badge>
        </div>

        <div className="flex gap-2">
          <Button onClick={loadPlantStatus} size="sm" variant="outline">
            <RotateCcw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </Card>

      {/* 3D Grid Visualization */}
      <Card className="p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">3D Plant Grid</h3>
          <div className="text-sm text-muted-foreground flex items-center">
            <Info className="mr-1 h-4 w-4" />
            Click on plants for details
          </div>
        </div>
        
        <div className="flex justify-center">
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            className="border rounded-lg cursor-pointer bg-white"
            style={{ maxWidth: '100%', height: 'auto' }}
          />
        </div>
      </Card>

      {/* Selected Plant Details */}
      {selectedPlant && (
        <Card className="p-4">
          <h3 className="text-lg font-semibold mb-4">Plant Details</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <strong>Plant ID:</strong> {selectedPlant.plant_id}
            </div>
            <div>
              <strong>Health Status:</strong> {selectedPlant.current_health}
            </div>
            <div>
              <strong>Position:</strong> Row {selectedPlant.position.row}, Column {selectedPlant.position.column}
            </div>
            <div>
              <strong>Greenhouse:</strong> {selectedPlant.position.greenhouse}
            </div>
            <div>
              <strong>Confidence:</strong> {(selectedPlant.confidence * 100).toFixed(1)}%
            </div>
            <div>
              <strong>Last Scan:</strong> {
                selectedPlant.last_scan_date 
                  ? new Date(selectedPlant.last_scan_date).toLocaleDateString()
                  : 'No scans yet'
              }
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};