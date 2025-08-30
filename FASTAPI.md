# FastAPI Endpoint for WhisperX

## Overview

A new FastAPI endpoint has been added to WhisperX that allows transcription via HTTP API calls. This provides a RESTful interface for integrating WhisperX into web applications, microservices, and other systems.

## Features

- **RESTful API**: HTTP endpoints for transcription
- **File Upload**: Support for audio file uploads
- **All WhisperX Features**: Alignment, diarization, multiple languages
- **Interactive Docs**: Swagger UI at `/docs`
- **Health Checks**: Server status monitoring
- **Validation**: Comprehensive input validation
- **Error Handling**: Detailed error messages
- **CORS Support**: Cross-origin request support

## Quick Start

1. **Start the server**:
   ```bash
   whisperx --serve
   ```

2. **Access the API**:
   - Health check: `GET http://localhost:8000/health`
   - Interactive docs: `http://localhost:8000/docs`
   - Transcribe: `POST http://localhost:8000/transcribe`

## Implementation Details

### New Files Added
- `whisperx/api.py`: FastAPI server implementation
- `examples/fastapi_example.py`: Usage example script

### Modified Files
- `pyproject.toml`: Added FastAPI dependencies
- `whisperx/__init__.py`: Added transcribe_audio export
- `whisperx/__main__.py`: Added --serve CLI option  
- `whisperx/transcribe.py`: Added reusable transcribe_audio function

### Dependencies Added
- `fastapi>=0.104.0`: Web framework
- `uvicorn>=0.24.0`: ASGI server
- `python-multipart>=0.0.6`: File upload support

## API Endpoints

### Health Check
```http
GET /health
```
Returns server status and device information.

### Supported Languages  
```http
GET /languages
```
Returns list of supported languages.

### Transcribe Audio
```http
POST /transcribe
```
Upload and transcribe audio files with full WhisperX feature support.

## Usage Examples

### Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Transcribe audio
with open("audio.wav", "rb") as f:
    files = {"audio_file": f}
    data = {"model": "base", "language": "en"}
    response = requests.post("http://localhost:8000/transcribe", 
                           files=files, data=data)
    result = response.json()
    print(result["text"])
```

### curl
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio_file=@audio.wav" \
  -F "model=base" \
  -F "language=en"
```

## CLI Integration

The FastAPI server is integrated into the main WhisperX CLI:

```bash
# Regular transcription (unchanged)
whisperx audio.wav

# Start API server
whisperx --serve
whisperx --serve --port 8080
whisperx --serve --host 0.0.0.0
```

## Architecture

The implementation follows these principles:
- **Minimal Changes**: Reuses existing transcription logic
- **Backward Compatibility**: Original CLI functionality unchanged
- **Lazy Loading**: Heavy dependencies loaded only when needed
- **Error Tolerance**: Graceful handling of missing dependencies
- **Clean Separation**: API code isolated in separate module

## Testing

Comprehensive tests verify:
- Server startup and health checks
- API endpoint functionality  
- Parameter validation
- Error handling
- CLI integration
- Documentation availability

## Deployment

For production use:
```bash
# Direct uvicorn
uvicorn whisperx.api:app --host 0.0.0.0 --port 8000 --workers 4

# With WhisperX CLI
whisperx --serve --host 0.0.0.0 --port 8000
```

Consider adding reverse proxy, SSL, authentication, and monitoring for production deployments.