# Copyright (c) 2025 Stephen G. Pope
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.



import os
import whisper
import srt
from datetime import timedelta
from services.file_management import download_file
import logging
from config import LOCAL_STORAGE_PATH, HF_TOKEN, WHISPER_CACHE_DIR

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def _assign_speakers_to_segments(segments, diarization_segments):
    """Assign speaker to each whisper segment by overlap with diarization segments."""
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        best_speaker = None
        best_overlap = 0.0
        for (d_start, d_end, speaker) in diarization_segments:
            overlap_start = max(seg_start, d_start)
            overlap_end = min(seg_end, d_end)
            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker
        seg["speaker"] = best_speaker if best_speaker else "UNKNOWN"


def _load_audio_for_pyannote(audio_path):
    """
    Load audio file into waveform + sample_rate for pyannote without using torchaudio/torchcodec.

    Strategy:
    - For non-WAV inputs (mp4, mkv, etc.) use ffmpeg to create a temporary 16 kHz mono WAV.
    - Use soundfile to read the WAV into a float32 numpy array.
    - Convert to torch.Tensor with shape (channels, time).

    Returns dict: {"waveform": (C, T) float32 tensor, "sample_rate": int}.
    """
    import subprocess
    import tempfile
    import torch
    import soundfile as sf
    import numpy as np

    ext = (os.path.splitext(audio_path)[1] or "").lower()
    wav_path = None

    if ext == ".wav":
        wav_path = audio_path
    else:
        # Normalize all other formats to 16 kHz mono WAV via ffmpeg.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=LOCAL_STORAGE_PATH) as f:
            wav_path = f.name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                wav_path,
            ],
            check=True,
            capture_output=True,
        )

    try:
        # soundfile returns shape (frames, channels); we transpose to (channels, frames)
        data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        # data: (frames, channels) -> (channels, frames)
        data = np.transpose(data, (1, 0))
        waveform = torch.from_numpy(data)
    finally:
        # Only remove temporary file, never the original input WAV.
        if ext != ".wav" and wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

    return {"waveform": waveform, "sample_rate": sample_rate}


def _run_diarization(audio_path, min_speakers=None, max_speakers=None):
    """Run pyannote diarization and return list of (start, end, speaker) tuples.
    Loads audio in-memory to avoid torchcodec/AudioDecoder (not installed in container)."""
    from pyannote.audio import Pipeline

    audio = _load_audio_for_pyannote(audio_path)
    pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        token=HF_TOKEN,
    )
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    diarization = pipeline(audio, **kwargs)
    # Newer pyannote returns DiarizeOutput (no itertracks); older returns Annotation (has itertracks)
    annotation = getattr(diarization, "speaker_diarization", None) or getattr(diarization, "annotation", None) or diarization
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    return segments


def process_transcribe_media(media_url, task, include_text, include_srt, include_segments, word_timestamps, response_type, language, job_id, words_per_line=None, diarize=False, min_speakers=None, max_speakers=None, model_size=None, progress_callback_url=None):
    """Transcribe or translate media and return the transcript/translation, SRT or VTT file path."""
    logger.info(f"Starting {task} for media URL: {media_url}")
    input_filename = download_file(media_url, os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_input"))
    logger.info(f"Downloaded media to local file: {input_filename}")

    try:
        _model = "large-v2" if task == "translate" else (model_size or "medium")
        model = whisper.load_model(_model, download_root=WHISPER_CACHE_DIR)
        logger.info(f"Loaded Whisper {_model} model")

        # Configure transcription/translation options
        options = {
            "task": task,
            "word_timestamps": word_timestamps,
            "verbose": False
        }

        # Add language specification if provided
        if language:
            options["language"] = language

        result = model.transcribe(input_filename, **options)

        if diarize and HF_TOKEN:
            logger.info("Running speaker diarization")
            diarization_segments = _run_diarization(input_filename, min_speakers, max_speakers)
            _assign_speakers_to_segments(result["segments"], diarization_segments)
        elif diarize and not HF_TOKEN:
            logger.warning("diarize=true but HF_TOKEN not set; skipping diarization")
        
        # For translation task, the result['text'] will be in English
        text = None
        srt_text = None
        segments_json = None

        logger.info(f"Generated {task} output")

        if include_text is True:
            text = result['text']

        if include_srt is True:
            srt_subtitles = []
            subtitle_index = 1
            
            if words_per_line and words_per_line > 0:
                # Collect all words, timings, and speakers (if diarized)
                all_words = []
                word_timings = []
                word_speakers = []
                
                for segment in result['segments']:
                    words = segment['text'].strip().split()
                    segment_start = segment['start']
                    segment_end = segment['end']
                    speaker = segment.get('speaker')
                    
                    # Calculate timing for each word
                    if words:
                        duration_per_word = (segment_end - segment_start) / len(words)
                        for i, word in enumerate(words):
                            word_start = segment_start + (i * duration_per_word)
                            word_end = word_start + duration_per_word
                            all_words.append(word)
                            word_timings.append((word_start, word_end))
                            word_speakers.append(speaker)
                
                # Process words in chunks of words_per_line
                current_word = 0
                while current_word < len(all_words):
                    # Get the next chunk of words
                    chunk = all_words[current_word:current_word + words_per_line]
                    
                    # Calculate timing for this chunk
                    chunk_start = word_timings[current_word][0]
                    chunk_end = word_timings[min(current_word + len(chunk) - 1, len(word_timings) - 1)][1]
                    
                    # Speaker: use first word's speaker when diarized
                    chunk_text = ' '.join(chunk)
                    if word_speakers and word_speakers[current_word]:
                        chunk_text = f"[{word_speakers[current_word]}] {chunk_text}"
                    
                    # Create the subtitle
                    srt_subtitles.append(srt.Subtitle(
                        subtitle_index,
                        timedelta(seconds=chunk_start),
                        timedelta(seconds=chunk_end),
                        chunk_text
                    ))
                    subtitle_index += 1
                    current_word += words_per_line
            else:
                # Original behavior - one subtitle per segment
                for segment in result['segments']:
                    start = timedelta(seconds=segment['start'])
                    end = timedelta(seconds=segment['end'])
                    segment_text = segment['text'].strip()
                    if segment.get('speaker'):
                        segment_text = f"[{segment['speaker']}] {segment_text}"
                    srt_subtitles.append(srt.Subtitle(subtitle_index, start, end, segment_text))
                    subtitle_index += 1
            
            srt_text = srt.compose(srt_subtitles)

        if include_segments is True:
            segments_json = result['segments']

        os.remove(input_filename)
        logger.info(f"Removed local file: {input_filename}")
        logger.info(f"{task.capitalize()} successful, output type: {response_type}")

        if response_type == "direct":
            return text, srt_text, segments_json
        else:
            
            if include_text is True:
                text_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.txt")
                with open(text_filename, 'w') as f:
                    f.write(text)
            else:
                text_file = None
            
            if include_srt is True:
                srt_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.srt")
                with open(srt_filename, 'w') as f:
                    f.write(srt_text)
            else:
                srt_filename = None

            if include_segments is True:
                segments_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.json")
                with open(segments_filename, 'w') as f:
                    f.write(str(segments_json))
            else:
                segments_filename = None

            return text_filename, srt_filename, segments_filename 

    except Exception as e:
        logger.error(f"{task.capitalize()} failed: {str(e)}")
        raise