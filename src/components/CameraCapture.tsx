import { useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Camera as CameraIcon, Image as ImageIcon, Upload } from 'lucide-react';
import { toast } from 'sonner';

interface CameraCaptureProps {
  onImageCapture: (imageData: string) => void;
}

export const CameraCapture = ({ onImageCapture }: CameraCaptureProps) => {
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      if (result) {
        onImageCapture(result);
      }
    };
    reader.onerror = () => {
      toast.error('Failed to read image. Please try again.');
    };
    reader.readAsDataURL(file);
  };

  const triggerCamera = () => {
    cameraInputRef.current?.click();
  };

  const triggerGallery = () => {
    galleryInputRef.current?.click();
  };

  return (
    <div className="flex flex-col gap-4 w-full">
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileSelect}
        className="hidden"
      />
      
      <input
        ref={galleryInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />
      
      <Button 
        onClick={triggerCamera}
        className="h-16 text-lg font-semibold bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity shadow-medium"
      >
        <CameraIcon className="mr-2 h-6 w-6" />
        Take Photo
      </Button>
      
      <Button 
        onClick={triggerGallery}
        variant="outline"
        className="h-16 text-lg font-semibold border-2 border-primary text-primary hover:bg-primary hover:text-primary-foreground transition-all"
      >
        <Upload className="mr-2 h-6 w-6" />
        Upload Image
      </Button>
    </div>
  );
};
