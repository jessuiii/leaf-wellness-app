import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2, CheckCircle, AlertTriangle, ArrowLeft, Clock, Leaf, MapPin } from 'lucide-react';
import { toast } from 'sonner';
import { apiEndpoints } from '@/config/api';

interface LeafAnalysisProps {
  imageData: string;
  imageFile?: File;
  onBack: () => void;
  onAnalysisComplete: (result: AnalysisResult) => void;
}

export interface AnalysisResult {
  isHealthy: boolean;
  confidence: number;
  disease?: string;
  recommendations?: string[];
  timestamp: number;
  imageData: string;
  plantId?: string;
  scanHistory?: ScanHistoryItem[];
}

export interface ScanHistoryItem {
  timestamp: string;
  disease: string;
  confidence: number;
  imageUrl?: string;
}

export interface PlantTwin {
  plant_id: string;
  current_health: string;
  last_scan_date?: string;
  visual_status: string;
  position: {
    row: string;
    column: number;
    greenhouse: string;
  };
}

export const LeafAnalysis = ({ imageData, imageFile, onBack, onAnalysisComplete }: LeafAnalysisProps) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [plantId, setPlantId] = useState<string>('');
  const [availablePlants, setAvailablePlants] = useState<PlantTwin[]>([]);
  const [isLoadingPlants, setIsLoadingPlants] = useState(false);
  const [showCreatePlant, setShowCreatePlant] = useState(false);
  const [newPlantData, setNewPlantData] = useState({
    row: '',
    column: '',
    greenhouse: 'Greenhouse_1',
    location: ''
  });

  // Load available plants on component mount
  useEffect(() => {
    loadAvailablePlants();
  }, []);

  const loadAvailablePlants = async () => {
    setIsLoadingPlants(true);
    try {
      const response = await fetch(apiEndpoints.allPlants());
      if (response.ok) {
        const data = await response.json();
        setAvailablePlants(data.plants || []);
      } else {
        console.log('No existing plants found or digital twin not available');
      }
    } catch (error) {
      console.log('Could not load plants:', error);
    } finally {
      setIsLoadingPlants(false);
    }
  };

  const createNewPlant = async () => {
    if (!plantId || !newPlantData.row || !newPlantData.column) {
      toast.error('Please fill in all required fields');
      return;
    }

    try {
      const response = await fetch(apiEndpoints.createPlant(plantId), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          plant_id: plantId,
          position: {
            row: newPlantData.row,
            column: parseInt(newPlantData.column),
            greenhouse: newPlantData.greenhouse
          },
          location: newPlantData.location || `${newPlantData.greenhouse}, Row ${newPlantData.row}, Position ${newPlantData.column}`
        })
      });

      if (response.ok) {
        toast.success('Plant created successfully');
        setShowCreatePlant(false);
        loadAvailablePlants();
      } else {
        const error = await response.json();
        toast.error(`Failed to create plant: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      toast.error('Network error while creating plant');
    }
  };

  const analyzeLeaf = async () => {
    if (!plantId.trim()) {
      toast.error('Please enter or select a Plant ID');
      return;
    }

    setIsAnalyzing(true);
    
    try {
      let response;
      
      if (imageFile) {
        // Use file upload endpoint
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('plant_id', plantId);
        
        response = await fetch(apiEndpoints.predictUpload(), {
          method: 'POST',
          body: formData
        });
      } else {
        // Use base64 endpoint (legacy)
        response = await fetch(apiEndpoints.predict(), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image: imageData,
            plant_id: plantId
          })
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      
      // Map API response to AnalysisResult format
      const analysisResult: AnalysisResult = {
        isHealthy: data.is_healthy,
        confidence: data.confidence,
        disease: data.disease,
        recommendations: data.recommendations || [],
        timestamp: data.timestamp,
        imageData: imageData,
        plantId: data.plant_id || plantId,
        scanHistory: data.scan_history || []
      };

      setResult(analysisResult);
      onAnalysisComplete(analysisResult);

      // Show success notification
      toast.success(
        data.is_healthy 
          ? '✅ Plant is healthy!' 
          : `⚠️ ${data.disease} detected (${(data.confidence * 100).toFixed(1)}% confidence)`
      );

    } catch (error: any) {
      console.error('Analysis error:', error);
      toast.error(error.message || 'Failed to analyze leaf. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Back Button */}
      <Button 
        variant="outline" 
        onClick={onBack}
        className="mb-4"
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to Camera
      </Button>

      {/* Plant Selection Section */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Leaf className="mr-2 h-5 w-5" />
          Plant Identification
        </h3>
        
        <div className="space-y-4">
          {/* Plant ID Input/Selection */}
          <div className="space-y-2">
            <Label htmlFor="plantId">Plant ID</Label>
            <div className="flex gap-2">
              <Input
                id="plantId"
                value={plantId}
                onChange={(e) => setPlantId(e.target.value)}
                placeholder="Enter plant ID (e.g., plant_A1)"
                className="flex-1"
              />
              <Button
                variant="outline"
                onClick={() => setShowCreatePlant(!showCreatePlant)}
                disabled={!plantId.trim()}
              >
                New Plant
              </Button>
            </div>
          </div>

          {/* Existing Plants Dropdown */}
          {availablePlants.length > 0 && (
            <div className="space-y-2">
              <Label>Or select existing plant:</Label>
              <Select value={plantId} onValueChange={setPlantId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a plant..." />
                </SelectTrigger>
                <SelectContent>
                  {availablePlants.map((plant) => (
                    <SelectItem key={plant.plant_id} value={plant.plant_id}>
                      <div className="flex items-center justify-between w-full">
                        <span>{plant.plant_id}</span>
                        <span className="ml-2 text-sm text-muted-foreground">
                          {plant.position.greenhouse}, Row {plant.position.row}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Create New Plant Form */}
          {showCreatePlant && (
            <Card className="p-4 bg-muted/50">
              <h4 className="font-medium mb-3">Create New Plant</h4>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Row</Label>
                  <Input
                    value={newPlantData.row}
                    onChange={(e) => setNewPlantData(prev => ({ ...prev, row: e.target.value.toUpperCase() }))}
                    placeholder="A, B, C..."
                    maxLength={1}
                  />
                </div>
                <div>
                  <Label>Column</Label>
                  <Input
                    type="number"
                    value={newPlantData.column}
                    onChange={(e) => setNewPlantData(prev => ({ ...prev, column: e.target.value }))}
                    placeholder="1, 2, 3..."
                    min={1}
                  />
                </div>
                <div className="col-span-2">
                  <Label>Greenhouse</Label>
                  <Select 
                    value={newPlantData.greenhouse} 
                    onValueChange={(value) => setNewPlantData(prev => ({ ...prev, greenhouse: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Greenhouse_1">Greenhouse 1</SelectItem>
                      <SelectItem value="Greenhouse_2">Greenhouse 2</SelectItem>
                      <SelectItem value="Outdoor">Outdoor Plot</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="col-span-2">
                  <Label>Location Description (Optional)</Label>
                  <Input
                    value={newPlantData.location}
                    onChange={(e) => setNewPlantData(prev => ({ ...prev, location: e.target.value }))}
                    placeholder="Additional location details..."
                  />
                </div>
              </div>
              <div className="flex gap-2 mt-3">
                <Button onClick={createNewPlant} size="sm">
                  Create Plant
                </Button>
                <Button variant="outline" onClick={() => setShowCreatePlant(false)} size="sm">
                  Cancel
                </Button>
              </div>
            </Card>
          )}
        </div>
      </Card>

      {/* Image Preview */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">Leaf Image</h3>
        <div className="flex justify-center">
          <img 
            src={imageData} 
            alt="Captured leaf" 
            className="max-w-full h-auto max-h-64 rounded-lg border"
          />
        </div>
      </Card>

      {/* Analysis Section */}
      <Card className="p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Disease Analysis</h3>
          {plantId && (
            <div className="flex items-center text-sm text-muted-foreground">
              <MapPin className="mr-1 h-4 w-4" />
              {plantId}
            </div>
          )}
        </div>

        <Button 
          onClick={analyzeLeaf} 
          disabled={isAnalyzing || !plantId.trim()}
          className="w-full mb-4"
          size="lg"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing Leaf...
            </>
          ) : (
            'Analyze Leaf Health'
          )}
        </Button>

        {/* Analysis Results */}
        {result && (
          <div className="space-y-4">
            {/* Health Status */}
            <div className={`p-4 rounded-lg border ${
              result.isHealthy 
                ? 'bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800' 
                : 'bg-red-50 border-red-200 dark:bg-red-950 dark:border-red-800'
            }`}>
              <div className="flex items-center">
                {result.isHealthy ? (
                  <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
                )}
                <div>
                  <div className="font-semibold">
                    {result.isHealthy ? 'Healthy Plant' : result.disease}
                  </div>
                  <div className="text-sm opacity-75">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            {result.recommendations && result.recommendations.length > 0 && (
              <div>
                <h4 className="font-semibold mb-2">Recommendations:</h4>
                <ul className="space-y-1 text-sm">
                  {result.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start">
                      <span className="text-muted-foreground mr-2">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Scan History */}
            {result.scanHistory && result.scanHistory.length > 0 && (
              <div>
                <h4 className="font-semibold mb-2 flex items-center">
                  <Clock className="mr-2 h-4 w-4" />
                  Recent Scans
                </h4>
                <div className="space-y-2">
                  {result.scanHistory.slice(0, 3).map((scan, index) => (
                    <div key={index} className="flex justify-between items-center text-sm bg-muted/50 p-2 rounded">
                      <div>
                        <div className="font-medium">{scan.disease}</div>
                        <div className="text-muted-foreground">
                          {new Date(scan.timestamp).toLocaleDateString()}
                        </div>
                      </div>
                      <div className="text-right">
                        <div>{(scan.confidence * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Timestamp */}
            <div className="text-xs text-muted-foreground text-center">
              Analysis completed at {new Date(result.timestamp).toLocaleString()}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};