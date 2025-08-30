import argparse
import gc
import os
import warnings
from typing import Union, Optional, Dict, List, Tuple

import numpy as np
import torch

from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.types import AlignedTranscriptionResult, TranscriptionResult
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE, get_writer


def transcribe_audio(
    audio_input: Union[str, np.ndarray],
    model_name: str = "small",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    device_index: int = 0,
    compute_type: str = "float16",
    batch_size: int = 8,
    language: Optional[str] = None,
    task: str = "transcribe",
    no_align: bool = False,
    align_model: Optional[str] = None,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    diarize: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    diarize_model_name: str = "pyannote/speaker-diarization-3.1",
    return_speaker_embeddings: bool = False,
    hf_token: Optional[str] = None,
    model_dir: Optional[str] = None,
    model_cache_only: bool = False,
    vad_method: str = "pyannote",
    vad_onset: float = 0.500,
    vad_offset: float = 0.363,
    chunk_size: int = 30,
    print_progress: bool = False,
    verbose: bool = False,
    **asr_kwargs
) -> Dict:
    """
    Transcribe audio using WhisperX pipeline.
    
    Args:
        audio_input: Audio file path or numpy array
        model_name: Whisper model name (default: "small")
        device: Device for inference (default: "cuda" if available else "cpu")
        device_index: Device index for FasterWhisper inference
        compute_type: Compute type for computation ("float16", "float32", "int8")
        batch_size: Batch size for inference
        language: Language code (None for auto-detection)
        task: Task type ("transcribe" or "translate")
        no_align: Skip word-level alignment
        align_model: Alignment model name
        interpolate_method: Method for timestamp interpolation
        return_char_alignments: Return character-level alignments
        diarize: Apply speaker diarization
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        diarize_model_name: Diarization model name
        return_speaker_embeddings: Include speaker embeddings in output
        hf_token: Hugging Face token for gated models
        model_dir: Directory to save/load models
        model_cache_only: Use only cached models
        vad_method: VAD method ("pyannote" or "silero")
        vad_onset: VAD onset threshold
        vad_offset: VAD offset threshold
        chunk_size: Chunk size for VAD segments
        print_progress: Print progress messages
        verbose: Verbose output
        **asr_kwargs: Additional ASR options
        
    Returns:
        Dictionary containing transcription results
    """
    
    # Validate and process language
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    # Handle model language constraints
    if model_name.endswith(".en") and language != "en":
        if language is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{language}'; using English instead."
            )
        language = "en"
    
    align_language = language if language is not None else "en"
    
    # Handle task constraints
    if task == "translate":
        no_align = True
    
    # Validate speaker embeddings
    if return_speaker_embeddings and not diarize:
        warnings.warn("return_speaker_embeddings has no effect without diarize=True")
    
    # Default ASR options
    default_asr_options = {
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "temperatures": [0.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "suppress_tokens": [-1],
        "suppress_numerals": False,
    }
    
    # Merge with provided ASR options
    asr_options = {**default_asr_options, **asr_kwargs}
    
    # Load audio if it's a file path
    if isinstance(audio_input, str):
        audio = load_audio(audio_input)
        audio_path = audio_input
    else:
        audio = audio_input
        audio_path = "uploaded_audio"
    
    # Part 1: VAD & ASR
    if print_progress:
        print(">>Loading ASR model...")
    
    model = load_model(
        model_name,
        device=device,
        device_index=device_index,
        download_root=model_dir,
        compute_type=compute_type,
        language=language,
        asr_options=asr_options,
        vad_method=vad_method,
        vad_options={
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        },
        task=task,
        local_files_only=model_cache_only,
        threads=4,
    )

    if print_progress:
        print(">>Performing transcription...")
    
    result: TranscriptionResult = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=chunk_size,
        print_progress=print_progress,
        verbose=verbose,
    )

    # Unload Whisper and VAD
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Part 2: Align
    if not no_align:
        if print_progress:
            print(">>Loading alignment model...")
        
        align_model_obj, align_metadata = load_align_model(
            align_language, device, model_name=align_model
        )
        
        if align_model_obj is not None and len(result["segments"]) > 0:
            if result.get("language", "en") != align_metadata["language"]:
                if print_progress:
                    print(
                        f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language..."
                    )
                align_model_obj, align_metadata = load_align_model(
                    result["language"], device
                )
            
            if print_progress:
                print(">>Performing alignment...")
            
            result: AlignedTranscriptionResult = align(
                result["segments"],
                align_model_obj,
                align_metadata,
                audio,
                device,
                interpolate_method=interpolate_method,
                return_char_alignments=return_char_alignments,
                print_progress=print_progress,
            )

        # Unload align model
        del align_model_obj
        gc.collect()
        torch.cuda.empty_cache()

    # Part 3: Diarize
    if diarize:
        if hf_token is None:
            if print_progress:
                print(
                    "Warning: no hf_token provided, needs to be saved in environment variable, otherwise will throw error loading diarization model..."
                )
        
        if print_progress:
            print(">>Performing diarization...")
            print(">>Using model:", diarize_model_name)
        
        diarize_model = DiarizationPipeline(
            model_name=diarize_model_name, 
            use_auth_token=hf_token, 
            device=device
        )
        
        diarize_result = diarize_model(
            audio if isinstance(audio_input, np.ndarray) else audio_path,
            min_speakers=min_speakers, 
            max_speakers=max_speakers, 
            return_embeddings=return_speaker_embeddings
        )

        if return_speaker_embeddings:
            diarize_segments, speaker_embeddings = diarize_result
        else:
            diarize_segments = diarize_result
            speaker_embeddings = None

        result = assign_word_speakers(diarize_segments, result, speaker_embeddings)

    # Set final language
    result["language"] = align_language
    
    return result


def transcribe_task(args: dict, parser: argparse.ArgumentParser):
    """Transcription task to be called from CLI.

    Args:
        args: Dictionary of command-line arguments.
        parser: argparse.ArgumentParser object.
    """
    # fmt: off

    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("model_dir")
    model_cache_only: bool = args.pop("model_cache_only")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")

    # model_flush: bool = args.pop("model_flush")
    os.makedirs(output_dir, exist_ok=True)

    align_model: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    task: str = args.pop("task")
    if task == "translate":
        # translation cannot be aligned
        no_align = True

    return_char_alignments: bool = args.pop("return_char_alignments")

    hf_token: str = args.pop("hf_token")
    vad_method: str = args.pop("vad_method")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")

    chunk_size: int = args.pop("chunk_size")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")
    diarize_model_name: str = args.pop("diarize_model")
    print_progress: bool = args.pop("print_progress")
    return_speaker_embeddings: bool = args.pop("speaker_embeddings")

    if return_speaker_embeddings and not diarize:
        warnings.warn("--speaker_embeddings has no effect without --diarize")

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")

    if model_name.endswith(".en") and args["language"] != "en":
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{args['language']}'; using English instead."
            )
        args["language"] = "en"
    align_language = (
        args["language"] if args["language"] is not None else "en"
    )  # default to loading english if not specified

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 4
    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}

    # Part 1: VAD & ASR Loop
    results = []
    tmp_results = []
    # model = load_model(model_name, device=device, download_root=model_dir)
    model = load_model(
        model_name,
        device=device,
        device_index=device_index,
        download_root=model_dir,
        compute_type=compute_type,
        language=args["language"],
        asr_options=asr_options,
        vad_method=vad_method,
        vad_options={
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        },
        task=task,
        local_files_only=model_cache_only,
        threads=faster_whisper_threads,
    )

    for audio_path in args.pop("audio"):
        audio = load_audio(audio_path)
        # >> VAD & ASR
        print(">>Performing transcription...")
        result: TranscriptionResult = model.transcribe(
            audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            print_progress=print_progress,
            verbose=verbose,
        )
        results.append((result, audio_path))

    # Unload Whisper and VAD
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Part 2: Align Loop
    if not no_align:
        tmp_results = results
        results = []
        align_model, align_metadata = load_align_model(
            align_language, device, model_name=align_model
        )
        for result, audio_path in tmp_results:
            # >> Align
            if len(tmp_results) > 1:
                input_audio = audio_path
            else:
                # lazily load audio from part 1
                input_audio = audio

            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language", "en") != align_metadata["language"]:
                    # load new language
                    print(
                        f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language..."
                    )
                    align_model, align_metadata = load_align_model(
                        result["language"], device
                    )
                print(">>Performing alignment...")
                result: AlignedTranscriptionResult = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    input_audio,
                    device,
                    interpolate_method=interpolate_method,
                    return_char_alignments=return_char_alignments,
                    print_progress=print_progress,
                )

            results.append((result, audio_path))

        # Unload align model
        del align_model
        gc.collect()
        torch.cuda.empty_cache()

    # >> Diarize
    if diarize:
        if hf_token is None:
            print(
                "Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model..."
            )
        tmp_results = results
        print(">>Performing diarization...")
        print(">>Using model:", diarize_model_name)
        results = []
        diarize_model = DiarizationPipeline(model_name=diarize_model_name, use_auth_token=hf_token, device=device)
        for result, input_audio_path in tmp_results:
            diarize_result = diarize_model(
                input_audio_path, 
                min_speakers=min_speakers, 
                max_speakers=max_speakers, 
                return_embeddings=return_speaker_embeddings
            )

            if return_speaker_embeddings:
                diarize_segments, speaker_embeddings = diarize_result
            else:
                diarize_segments = diarize_result
                speaker_embeddings = None

            result = assign_word_speakers(diarize_segments, result, speaker_embeddings)
            results.append((result, input_audio_path))
    # >> Write
    for result, audio_path in results:
        result["language"] = align_language
        writer(result, audio_path, writer_args)
