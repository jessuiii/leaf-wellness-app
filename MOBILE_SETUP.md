# LeafGuard Mobile Setup Guide

## Overview
LeafGuard is now configured as a mobile app using Capacitor! You can run it on iOS and Android devices.

## Quick Start (Testing in Browser)
The app works in your browser right now with simulated camera access. Perfect for development and testing!

## Running on Physical Devices

### Prerequisites
- Node.js and npm installed
- For iOS: Mac with Xcode
- For Android: Android Studio

### Setup Steps

1. **Transfer to GitHub**
   - Click "Export to GitHub" in Lovable
   - Clone your repository locally
   ```bash
   git clone <YOUR_GIT_URL>
   cd <YOUR_PROJECT_NAME>
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Add Mobile Platforms**
   ```bash
   # For iOS (Mac only)
   npx cap add ios
   
   # For Android
   npx cap add android
   ```

4. **Build the Project**
   ```bash
   npm run build
   ```

5. **Sync with Native Projects**
   ```bash
   npx cap sync
   ```

6. **Run on Device/Emulator**
   ```bash
   # For Android
   npx cap run android
   
   # For iOS (Mac only)
   npx cap run ios
   ```

## Camera Permissions
The app is pre-configured with the necessary camera permissions:
- iOS: Camera and Photo Library access
- Android: Camera, Read/Write External Storage

## Backend API Integration
Currently using a mock API for demo purposes. To connect your real backend:

1. Open `src/components/LeafAnalysis.tsx`
2. Replace the mock API call with your actual endpoint:
   ```typescript
   const response = await fetch('YOUR_API_ENDPOINT', {
     method: 'POST',
     body: JSON.stringify({ image: imageData }),
     headers: { 'Content-Type': 'application/json' }
   });
   const data = await response.json();
   ```

## Features
- ✅ Camera capture
- ✅ Gallery selection
- ✅ Image preview
- ✅ AI disease detection
- ✅ Scan history
- ✅ Treatment recommendations
- ✅ Responsive mobile design

## Troubleshooting
- **Camera not working?** Make sure you've granted camera permissions
- **Build errors?** Run `npx cap sync` after any code changes
- **iOS build issues?** Open in Xcode and check signing certificates

## Development Workflow
1. Make code changes in Lovable or your IDE
2. Pull latest changes: `git pull`
3. Sync native projects: `npx cap sync`
4. Test on device/emulator

For more help, visit: https://capacitorjs.com/docs
