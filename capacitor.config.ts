import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'app.lovable.ea7323f627d342d69b2890c4a79d3150',
  appName: 'LeafGuard',
  webDir: 'dist',
  server: {
    url: 'https://ea7323f6-27d3-42d6-9b28-90c4a79d3150.lovableproject.com?forceHideBadge=true',
    cleartext: true
  },
  plugins: {
    Camera: {
      ios: {
        NSCameraUsageDescription: 'LeafGuard needs camera access to capture leaf photos for disease detection.',
        NSPhotoLibraryUsageDescription: 'LeafGuard needs photo library access to select leaf images for analysis.',
      },
      android: {
        permissions: ['camera', 'read_external_storage', 'write_external_storage']
      }
    }
  }
};

export default config;
