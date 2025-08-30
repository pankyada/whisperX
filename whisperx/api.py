"""
FastAPI server for WhisperX transcription service.
"""
import io
import tempfile
import os
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import transcription function only when needed
# from whisperx.transcribe import transcribe_audio
# from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE


# Create FastAPI app
app = FastAPI(
    title="WhisperX Transcription API",
    description="Fast and accurate speech transcription using WhisperX",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionResponse(BaseModel):
    """Response model for transcription results."""
    segments: List[dict]
    language: str
    text: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    device: str
    cuda_available: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
    except ImportError:
        cuda_available = False
        device = "cpu"
    
    return HealthResponse(
        status="healthy",
        device=device,
        cuda_available=cuda_available
    )


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages."""
    try:
        from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE
        return {
            "languages": list(LANGUAGES.keys()),
            "language_codes": list(TO_LANGUAGE_CODE.keys())
        }
    except ImportError:
        # Return a basic set if whisperx utils are not available
        return {
            "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh"],
            "language_codes": ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Japanese", "Chinese"]
        }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form("small", description="Whisper model name"),
    language: Optional[str] = Form(None, description="Language code (auto-detect if None)"),
    task: str = Form("transcribe", description="Task: 'transcribe' or 'translate'"),
    device: Optional[str] = Form(None, description="Device: 'cuda' or 'cpu' (auto-detect if None)"),
    batch_size: int = Form(8, description="Batch size for inference"),
    compute_type: str = Form("float16", description="Compute type: 'float16', 'float32', or 'int8'"),
    no_align: bool = Form(False, description="Skip word-level alignment"),
    diarize: bool = Form(False, description="Apply speaker diarization"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers"),
    return_char_alignments: bool = Form(False, description="Return character-level alignments"),
    print_progress: bool = Form(False, description="Print progress messages"),
):
    """
    Transcribe an audio file using WhisperX.
    
    Upload an audio file and get back a transcription with word-level timestamps.
    Supports various audio formats including WAV, MP3, FLAC, etc.
    """
    
    # Check if dependencies are available
    try:
        from whisperx.transcribe import transcribe_audio
        from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE
        import torch
    except ImportError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"WhisperX dependencies not available: {e}. Please install whisperx with all dependencies."
        )
    
    # Validate file
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Check file size (limit to 100MB)
    if audio_file.size and audio_file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 100MB")
    
    # Validate language
    if language and language not in LANGUAGES and language not in TO_LANGUAGE_CODE:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language: {language}. Use /languages endpoint to see supported languages"
        )
    
    # Validate task
    if task not in ["transcribe", "translate"]:
        raise HTTPException(status_code=400, detail="Task must be 'transcribe' or 'translate'")
    
    # Validate compute type
    if compute_type not in ["float16", "float32", "int8"]:
        raise HTTPException(status_code=400, detail="Compute type must be 'float16', 'float32', or 'int8'")
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Validate device
    if device not in ["cuda", "cpu"]:
        raise HTTPException(status_code=400, detail="Device must be 'cuda' or 'cpu'")
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA not available. Use device='cpu'")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
            temp_file.write(await audio_file.read())
            temp_file_path = temp_file.name
        
        try:
            # Perform transcription
            result = transcribe_audio(
                audio_input=temp_file_path,
                model_name=model,
                device=device,
                batch_size=batch_size,
                compute_type=compute_type,
                language=language,
                task=task,
                no_align=no_align,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_char_alignments=return_char_alignments,
                print_progress=print_progress,
                verbose=False
            )
            
            # Extract full text from segments
            full_text = " ".join(segment.get("text", "").strip() for segment in result.get("segments", []))
            
            return TranscriptionResponse(
                segments=result.get("segments", []),
                language=result.get("language", "unknown"),
                text=full_text
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)