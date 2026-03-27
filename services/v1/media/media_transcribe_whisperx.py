import os
import logging
from datetime import timedelta

import srt
import torch
import whisperx

from config import LOCAL_STORAGE_PATH, HF_TOKEN
from services.file_management import download_file
from services.v1.media.progress_callback import report_progress


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30  # report progress after each chunk
TRANSCRIBE_PCT = 90  # 0-90% transcribe, 90-100% diarize


def _get_device_and_compute_type() -> tuple[str, str]:
    """Select best available device and compute type for WhisperX."""
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def _build_srt_from_segments(segments: list[dict], words_per_line: int | None = None) -> str:
    """Compose SRT text from WhisperX segments, optionally chunked by word count."""
    subtitles: list[srt.Subtitle] = []
    subtitle_index = 1

    if words_per_line and words_per_line > 0:
        all_words: list[str] = []
        word_timings: list[tuple[float, float]] = []
        word_speakers: list[str | None] = []

        for segment in segments:
            words = segment.get("text", "").strip().split()
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
            speaker = segment.get("speaker")

            if not words:
                continue

            duration_per_word = (seg_end - seg_start) / len(words) if seg_end > seg_start else 0.0
            for i, word in enumerate(words):
                word_start = seg_start + i * duration_per_word
                word_end = word_start + duration_per_word
                all_words.append(word)
                word_timings.append((word_start, word_end))
                word_speakers.append(speaker)

        current = 0
        while current < len(all_words):
            chunk = all_words[current : current + words_per_line]
            if not chunk:
                break

            chunk_start = word_timings[current][0]
            chunk_end = word_timings[min(current + len(chunk) - 1, len(word_timings) - 1)][1]

            chunk_text = " ".join(chunk)
            if word_speakers and word_speakers[current]:
                chunk_text = f"[{word_speakers[current]}] {chunk_text}"

            subtitles.append(
                srt.Subtitle(
                    subtitle_index,
                    timedelta(seconds=chunk_start),
                    timedelta(seconds=chunk_end),
                    chunk_text,
                )
            )
            subtitle_index += 1
            current += words_per_line
    else:
        for segment in segments:
            start = timedelta(seconds=float(segment.get("start", 0.0)))
            end = timedelta(seconds=float(segment.get("end", 0.0)))
            text = (segment.get("text") or "").strip()
            speaker = segment.get("speaker")
            if speaker:
                text = f"[{speaker}] {text}"
            if not text:
                continue
            subtitles.append(srt.Subtitle(subtitle_index, start, end, text))
            subtitle_index += 1

    return srt.compose(subtitles)


def process_transcribe_media_whisperx(
    media_url: str,
    task: str,
    include_text: bool,
    include_srt: bool,
    include_segments: bool,
    response_type: str,
    language: str | None,
    job_id: str,
    words_per_line: int | None = None,
    diarize: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    model_size: str | None = None,
    progress_callback_url: str | None = None,
):
    """
    Transcribe media using WhisperX (alignment + optional diarization).

    Returns:
        If response_type == "direct": (text, srt_text, segments_json)
        Else: (text_path, srt_path, segments_path)
    """
    logger.info("WhisperX: starting %s for media URL: %s", task, media_url)
    report_progress(progress_callback_url, "download")
    input_path = download_file(media_url, os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_input"))
    logger.info("WhisperX: downloaded media to %s", input_path)

    device, compute_type = _get_device_and_compute_type()
    logger.info("WhisperX: using device=%s compute_type=%s", device, compute_type)

    try:
        audio = whisperx.load_audio(input_path)
        total_seconds = len(audio) / float(SAMPLE_RATE) if hasattr(audio, "__len__") else 0.0
        report_progress(
            progress_callback_url, "transcribe",
            current_seconds=0.0, total_seconds=total_seconds if total_seconds > 0 else None,
            progress_pct=0,
        )
        _model = "large-v2" if task == "translate" else (model_size or "medium")
        load_kwargs = {"device": device, "compute_type": compute_type}
        if language:
            load_kwargs["language"] = language.strip().lower()
        model = whisperx.load_model(_model, **load_kwargs)

        chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
        if total_seconds <= CHUNK_SECONDS or len(audio) <= chunk_samples:
            asr_result = model.transcribe(audio)
            if total_seconds > 0:
                report_progress(
                    progress_callback_url, "transcribe",
                    current_seconds=total_seconds, total_seconds=total_seconds,
                    progress_pct=TRANSCRIBE_PCT,
                )
        else:
            all_segments = []
            detected_lang = None
            for start_sample in range(0, len(audio), chunk_samples):
                chunk = audio[start_sample : start_sample + chunk_samples]
                if len(chunk) == 0:
                    break
                chunk_start_sec = start_sample / float(SAMPLE_RATE)
                chunk_result = model.transcribe(chunk)
                for seg in chunk_result.get("segments", []):
                    seg = dict(seg)
                    seg["start"] = seg.get("start", 0) + chunk_start_sec
                    seg["end"] = seg.get("end", 0) + chunk_start_sec
                    all_segments.append(seg)
                if detected_lang is None and chunk_result.get("language"):
                    detected_lang = chunk_result["language"]
                chunk_end_sec = min(chunk_start_sec + len(chunk) / float(SAMPLE_RATE), total_seconds)
                pct = round((chunk_end_sec / total_seconds) * TRANSCRIBE_PCT) if total_seconds > 0 else 0
                report_progress(
                    progress_callback_url, "transcribe",
                    current_seconds=chunk_end_sec, total_seconds=total_seconds,
                    progress_pct=min(pct, TRANSCRIBE_PCT),
                )
            all_segments.sort(key=lambda s: (s.get("start", 0), s.get("end", 0)))
            asr_result = {"segments": all_segments, "language": detected_lang or language}

        align_lang = language or asr_result.get("language")
        model_a, metadata = whisperx.load_align_model(language_code=align_lang, device=device)
        aligned = whisperx.align(
            asr_result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        segments = aligned["segments"]

        if diarize and HF_TOKEN:
            report_progress(progress_callback_url, "diarize", progress_pct=TRANSCRIBE_PCT)
            logger.info("WhisperX: running diarization with pyannote (token provided)")
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
            diarize_kwargs: dict = {}
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            aligned = whisperx.assign_word_speakers(diarize_segments, aligned)
            segments = aligned["segments"]
        elif diarize and not HF_TOKEN:
            logger.warning("WhisperX: diarize=true but HF_TOKEN not set; skipping diarization")

        text = None
        srt_text = None
        segments_json = None

        if include_text:
            text = " ".join((seg.get("text") or "").strip() for seg in segments if seg.get("text"))

        if include_srt:
            srt_text = _build_srt_from_segments(segments, words_per_line)

        if include_segments:
            segments_json = segments

        os.remove(input_path)
        logger.info("WhisperX: removed local file %s", input_path)
        report_progress(progress_callback_url, "done", progress_pct=100)

        if response_type == "direct":
            return text, srt_text, segments_json

        text_path = srt_path = segments_path = None

        if include_text:
            text_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_whisperx.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text or "")

        if include_srt:
            srt_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_whisperx.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_text or "")

        if include_segments:
            segments_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_whisperx.json")
            import json as _json

            with open(segments_path, "w", encoding="utf-8") as f:
                _json.dump(segments_json or [], f)

        return text_path, srt_path, segments_path

    except Exception as exc:
        logger.error("WhisperX transcription failed: %s", exc)
        raise

