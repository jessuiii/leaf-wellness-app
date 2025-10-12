import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import { Button } from '@/components/ui/button';
import { Camera as CameraIcon, Image as ImageIcon } from 'lucide-react';
import { toast } from 'sonner';

interface CameraCaptureProps {
  onImageCapture: (imageData: string) => void;
}

export const CameraCapture = ({ onImageCapture }: CameraCaptureProps) => {
  const takePhoto = async () => {
    try {
      const image = await Camera.getPhoto({
        quality: 90,
        allowEditing: false,
        resultType: CameraResultType.DataUrl,
        source: CameraSource.Camera,
      });

      if (image.dataUrl) {
        onImageCapture(image.dataUrl);
      }
    } catch (error) {
      console.error('Error taking photo:', error);
      toast.error('Failed to capture photo. Please try again.');
    }
  };

  const selectFromGallery = async () => {
    try {
      const image = await Camera.getPhoto({
        quality: 90,
        allowEditing: false,
        resultType: CameraResultType.DataUrl,
        source: CameraSource.Photos,
      });

      if (image.dataUrl) {
        onImageCapture(image.dataUrl);
      }
    } catch (error) {
      console.error('Error selecting photo:', error);
      toast.error('Failed to select photo. Please try again.');
    }
  };

  return (
    <div className="flex flex-col gap-4 w-full">
      <Button 
        onClick={takePhoto}
        className="h-16 text-lg font-semibold bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity shadow-medium"
      >
        <CameraIcon className="mr-2 h-6 w-6" />
        Take Photo
      </Button>
      
      <Button 
        onClick={selectFromGallery}
        variant="outline"
        className="h-16 text-lg font-semibold border-2 border-primary text-primary hover:bg-primary hover:text-primary-foreground transition-all"
      >
        <ImageIcon className="mr-2 h-6 w-6" />
        Choose from Gallery
      </Button>
    </div>
  );
};
