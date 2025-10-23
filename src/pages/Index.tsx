import { useState, useEffect } from 'react';
import { CameraCapture } from '@/components/CameraCapture';
import { LeafAnalysis, AnalysisResult } from '@/components/LeafAnalysis';
import { ScanHistory } from '@/components/ScanHistory';
import { PlantViewer3D } from '@/components/PlantViewer3D';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Leaf, Scan, History, Boxes } from 'lucide-react';

const Index = () => {
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [capturedFile, setCapturedFile] = useState<File | null>(null);
  const [scanHistory, setScanHistory] = useState<AnalysisResult[]>([]);

  useEffect(() => {
    // Load scan history from localStorage
    const savedHistory = localStorage.getItem('leafGuardHistory');
    if (savedHistory) {
      setScanHistory(JSON.parse(savedHistory));
    }
  }, []);

  const handleImageCapture = (imageData: string, imageFile?: File) => {
    setCapturedImage(imageData);
    setCapturedFile(imageFile || null);
  };

  const handleAnalysisComplete = (result: AnalysisResult) => {
    const newHistory = [result, ...scanHistory].slice(0, 20); // Keep last 20 scans
    setScanHistory(newHistory);
    localStorage.setItem('leafGuardHistory', JSON.stringify(newHistory));
  };

  const handleBack = () => {
    setCapturedImage(null);
    setCapturedFile(null);
  };
    setCapturedImage(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-secondary/20 to-background">
      <div className="container max-w-2xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-primary to-accent mb-4 shadow-medium">
            <Leaf className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent mb-2">
            LeafGuard
          </h1>
          <p className="text-muted-foreground text-lg">
            AI-Powered Leaf Disease Detection
          </p>
        </div>

        {!capturedImage ? (
          <Tabs defaultValue="scan" className="w-full">
            <TabsList className="grid w-full grid-cols-3 mb-8">
              <TabsTrigger value="scan" className="text-lg flex items-center">
                <Scan className="mr-2 h-4 w-4" />
                Scan
              </TabsTrigger>
              <TabsTrigger value="history" className="text-lg flex items-center">
                <History className="mr-2 h-4 w-4" />
                History
              </TabsTrigger>
              <TabsTrigger value="3d-view" className="text-lg flex items-center">
                <Boxes className="mr-2 h-4 w-4" />
                3D View
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="scan" className="space-y-6">
              <div className="bg-card rounded-2xl p-8 shadow-medium">
                <h2 className="text-2xl font-bold mb-6 text-center">
                  Capture a Leaf Photo
                </h2>
                <CameraCapture onImageCapture={handleImageCapture} />
              </div>
              
              <div className="bg-card rounded-2xl p-6 shadow-soft">
                <h3 className="font-semibold mb-3">How it works:</h3>
                <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                  <li>Select or create a plant ID for tracking</li>
                  <li>Take a photo or select from gallery</li>
                  <li>Our AI analyzes the leaf and updates the digital twin</li>
                  <li>Get instant disease detection results</li>
                  <li>Follow recommendations for treatment</li>
                  <li>View plant history and 3D visualization</li>
                </ol>
              </div>
            </TabsContent>
            
            <TabsContent value="history">
              <div className="bg-card rounded-2xl p-6 shadow-medium">
                <ScanHistory history={scanHistory} />
              </div>
            </TabsContent>

            <TabsContent value="3d-view">
              <div className="bg-card rounded-2xl p-6 shadow-medium">
                <PlantViewer3D />
              </div>
            </TabsContent>
          </Tabs>
        ) : (
          <div className="bg-card rounded-2xl p-8 shadow-medium">
            <LeafAnalysis
              imageData={capturedImage}
              imageFile={capturedFile}
              onBack={handleBack}
              onAnalysisComplete={handleAnalysisComplete}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default Index;
