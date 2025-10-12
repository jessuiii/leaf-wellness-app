import { Card } from '@/components/ui/card';
import { CheckCircle, AlertTriangle } from 'lucide-react';
import { AnalysisResult } from './LeafAnalysis';

interface ScanHistoryProps {
  history: AnalysisResult[];
}

export const ScanHistory = ({ history }: ScanHistoryProps) => {
  if (history.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No scans yet. Start by capturing a leaf!</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Recent Scans</h2>
      
      <div className="grid gap-4">
        {history.map((scan, idx) => (
          <Card key={idx} className="overflow-hidden shadow-soft hover:shadow-medium transition-shadow">
            <div className="flex gap-4 p-4">
              <img 
                src={scan.imageData} 
                alt="Scanned leaf" 
                className="w-24 h-24 object-cover rounded-lg flex-shrink-0"
              />
              
              <div className="flex-1 min-w-0">
                <div className="flex items-start gap-2 mb-2">
                  {scan.isHealthy ? (
                    <CheckCircle className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <p className="font-semibold">
                      {scan.isHealthy ? 'Healthy' : scan.disease || 'Diseased'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {new Date(scan.timestamp).toLocaleDateString()} at {new Date(scan.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
                
                <div className="text-sm text-muted-foreground">
                  Confidence: {scan.confidence.toFixed(1)}%
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};
