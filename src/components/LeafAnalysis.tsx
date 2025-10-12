import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Loader2, CheckCircle, AlertTriangle, ArrowLeft } from 'lucide-react';
import { toast } from 'sonner';

interface LeafAnalysisProps {
  imageData: string;
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
}

export const LeafAnalysis = ({ imageData, onBack, onAnalysisComplete }: LeafAnalysisProps) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const analyzeLeaf = async () => {
    setIsAnalyzing(true);
    
    try {
      // TODO: Replace with actual API call to your backend
      // const response = await fetch('YOUR_API_ENDPOINT', {
      //   method: 'POST',
      //   body: JSON.stringify({ image: imageData }),
      //   headers: { 'Content-Type': 'application/json' }
      // });
      // const data = await response.json();
      
      // Simulated API response for demo
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockResult: AnalysisResult = {
        isHealthy: Math.random() > 0.5,
        confidence: 85 + Math.random() * 15,
        disease: Math.random() > 0.5 ? undefined : 'Early Blight',
        recommendations: [
          'Remove affected leaves',
          'Apply fungicide treatment',
          'Monitor nearby plants'
        ],
        timestamp: Date.now(),
        imageData
      };
      
      setResult(mockResult);
      onAnalysisComplete(mockResult);
      
      toast.success('Analysis complete!');
    } catch (error) {
      console.error('Analysis error:', error);
      toast.error('Failed to analyze leaf. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 w-full">
      <Button
        variant="ghost"
        onClick={onBack}
        className="self-start -ml-2"
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back
      </Button>

      <Card className="overflow-hidden shadow-medium">
        <img 
          src={imageData} 
          alt="Captured leaf" 
          className="w-full h-64 object-cover"
        />
      </Card>

      {!result && (
        <Button
          onClick={analyzeLeaf}
          disabled={isAnalyzing}
          className="h-14 text-lg font-semibold bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity shadow-medium"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            'Analyze Leaf'
          )}
        </Button>
      )}

      {result && (
        <Card className={`p-6 ${result.isHealthy ? 'bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950' : 'bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-950 dark:to-orange-950'} shadow-medium`}>
          <div className="flex items-start gap-4">
            {result.isHealthy ? (
              <CheckCircle className="h-12 w-12 text-primary flex-shrink-0" />
            ) : (
              <AlertTriangle className="h-12 w-12 text-destructive flex-shrink-0" />
            )}
            
            <div className="flex-1">
              <h3 className="text-2xl font-bold mb-2">
                {result.isHealthy ? 'Healthy Leaf' : 'Disease Detected'}
              </h3>
              
              <div className="space-y-2 text-sm">
                <p className="font-medium">
                  Confidence: {result.confidence.toFixed(1)}%
                </p>
                
                {result.disease && (
                  <p className="font-semibold text-lg text-destructive">
                    Disease: {result.disease}
                  </p>
                )}
                
                {result.recommendations && result.recommendations.length > 0 && (
                  <div className="mt-4">
                    <p className="font-semibold mb-2">Recommendations:</p>
                    <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                      {result.recommendations.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        </Card>
      )}

      {result && (
        <Button
          onClick={onBack}
          variant="outline"
          className="h-12 border-2"
        >
          Scan Another Leaf
        </Button>
      )}
    </div>
  );
};
