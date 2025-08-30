#!/usr/bin/env python3
"""
Example script demonstrating how to use the WhisperX FastAPI endpoint.
"""

import requests
import time
import subprocess
import sys
import os

def test_whisperx_api():
    """
    Example of how to use the WhisperX FastAPI endpoint.
    """
    
    print("WhisperX FastAPI Example")
    print("=" * 40)
    
    # Server URL
    base_url = "http://localhost:8000"
    
    # 1. Check if server is running
    print("\n1. Checking server health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✓ Server is healthy")
            print(f"   ✓ Device: {health['device']}")
            print(f"   ✓ CUDA Available: {health['cuda_available']}")
        else:
            print(f"   ✗ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Cannot connect to server: {e}")
        print("   ℹ  Start the server with: whisperx --serve")
        return False
    
    # 2. Get supported languages
    print("\n2. Getting supported languages...")
    try:
        response = requests.get(f"{base_url}/languages", timeout=5)
        if response.status_code == 200:
            languages = response.json()
            print(f"   ✓ {len(languages['languages'])} languages supported")
            print(f"   ✓ Sample: {', '.join(languages['languages'][:10])}")
        else:
            print(f"   ✗ Failed to get languages: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Error getting languages: {e}")
    
    # 3. Test transcription endpoint (without actual audio file)
    print("\n3. Testing transcription endpoint...")
    print("   (Note: This will fail gracefully as we don't have actual audio dependencies)")
    
    # Create a dummy "audio" file for demonstration
    dummy_audio = b"This is not real audio data, just for testing the API"
    
    try:
        files = {
            'audio_file': ('test_audio.wav', dummy_audio, 'audio/wav')
        }
        
        data = {
            'model': 'base',
            'language': 'en',
            'task': 'transcribe',
            'device': 'cpu',
            'no_align': True,  # Skip alignment to avoid more dependencies
            'print_progress': False
        }
        
        response = requests.post(f"{base_url}/transcribe", files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ Transcription successful!")
            print(f"   ✓ Language detected: {result.get('language', 'unknown')}")
            print(f"   ✓ Text: {result.get('text', 'No text')}")
            print(f"   ✓ Segments: {len(result.get('segments', []))}")
        elif response.status_code == 500:
            error = response.json()
            if 'dependencies not available' in error.get('detail', ''):
                print("   ✓ API correctly reports missing dependencies")
                print("   ℹ  This is expected when WhisperX dependencies are not fully installed")
            else:
                print(f"   ⚠  Server error: {error.get('detail', 'Unknown error')}")
        else:
            print(f"   ⚠  Unexpected response: {response.status_code}")
            print(f"   ⚠  Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Transcription request failed: {e}")
    
    # 4. Show API documentation info
    print("\n4. API Documentation:")
    print(f"   🌐 Interactive docs: {base_url}/docs")
    print(f"   🌐 OpenAPI schema: {base_url}/openapi.json")
    
    # 5. Usage examples
    print("\n5. Usage Examples:")
    print("""
   Python example:
   ```python
   import requests
   
   # Transcribe an audio file
   with open('audio.wav', 'rb') as f:
       files = {'audio_file': f}
       data = {'model': 'base', 'language': 'en'}
       response = requests.post('http://localhost:8000/transcribe', 
                               files=files, data=data)
       result = response.json()
       print(result['text'])
   ```
   
   curl example:
   ```bash
   curl -X POST "http://localhost:8000/transcribe" \\
     -F "audio_file=@audio.wav" \\
     -F "model=base" \\
     -F "language=en"
   ```
   """)
    
    return True


def start_server_if_needed():
    """
    Check if server is running, and offer to start it if not.
    """
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            return True
    except:
        pass
    
    print("WhisperX server is not running.")
    answer = input("Would you like to start it? (y/N): ").strip().lower()
    
    if answer in ['y', 'yes']:
        print("Starting WhisperX server...")
        print("(This will run in the background. Press Ctrl+C to stop both this script and the server)")
        
        try:
            # Start server in background
            process = subprocess.Popen([
                sys.executable, "-m", "whisperx", "--serve"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            print("Waiting for server to start...")
            for i in range(10):
                time.sleep(1)
                try:
                    response = requests.get("http://localhost:8000/health", timeout=1)
                    if response.status_code == 200:
                        print("Server started successfully!")
                        return True
                except:
                    continue
            
            print("Server failed to start within 10 seconds")
            process.terminate()
            return False
            
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    return False


def main():
    """Main function."""
    print("WhisperX FastAPI Example Script")
    print("This script demonstrates how to use the WhisperX API endpoint")
    print()
    
    # Check if server is running or start it
    if not start_server_if_needed():
        print("\nTo manually start the server, run:")
        print("  whisperx --serve")
        print("\nThen run this script again.")
        return 1
    
    # Run the API test
    success = test_whisperx_api()
    
    if success:
        print("\n" + "=" * 40)
        print("✅ WhisperX FastAPI endpoint is working!")
        print("=" * 40)
        return 0
    else:
        print("\n" + "=" * 40)
        print("❌ Some issues were encountered")
        print("=" * 40)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
        sys.exit(0)