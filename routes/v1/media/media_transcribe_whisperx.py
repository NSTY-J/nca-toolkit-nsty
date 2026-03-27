from flask import Blueprint

from app_utils import queue_task_wrapper, validate_payload
from services.authentication import authenticate
from services.cloud_storage import upload_file
from services.v1.media.media_transcribe_whisperx import process_transcribe_media_whisperx
import logging
import os


v1_media_transcribe_whisperx_bp = Blueprint("v1_media_transcribe_whisperx", __name__)
logger = logging.getLogger(__name__)


@v1_media_transcribe_whisperx_bp.route("/v1/media/transcribe/whisperx", methods=["POST"])
@authenticate
@validate_payload(
    {
        "type": "object",
        "properties": {
            "media_url": {"type": "string", "format": "uri"},
            "task": {"type": "string", "enum": ["transcribe", "translate"]},
            "include_text": {"type": "boolean"},
            "include_srt": {"type": "boolean"},
            "include_segments": {"type": "boolean"},
            "response_type": {"type": "string", "enum": ["direct", "cloud"]},
            "language": {"type": "string"},
            "webhook_url": {"type": "string", "format": "uri"},
            "id": {"type": "string"},
            "words_per_line": {"type": "integer", "minimum": 1},
            "diarize": {"type": "boolean"},
            "min_speakers": {"type": "integer", "minimum": 1},
            "max_speakers": {"type": "integer", "minimum": 1},
        },
        "required": ["media_url"],
        "additionalProperties": False,
    }
)
@queue_task_wrapper(bypass_queue=False)
def transcribe_whisperx(job_id, data):
    media_url = data["media_url"]
    task = data.get("task", "transcribe")
    include_text = data.get("include_text", True)
    include_srt = data.get("include_srt", False)
    include_segments = data.get("include_segments", False)
    response_type = data.get("response_type", "direct")
    language = data.get("language", None)
    webhook_url = data.get("webhook_url")
    _id = data.get("id")
    words_per_line = data.get("words_per_line", None)
    diarize = data.get("diarize", True)
    min_speakers = data.get("min_speakers", None)
    max_speakers = data.get("max_speakers", None)

    logger.info("Job %s (whisperx): Received transcription request for %s", job_id, media_url)

    try:
        result = process_transcribe_media_whisperx(
            media_url,
            task,
            include_text,
            include_srt,
            include_segments,
            response_type,
            language,
            job_id,
            words_per_line,
            diarize,
            min_speakers,
            max_speakers,
        )
        logger.info("Job %s (whisperx): Transcription process completed successfully", job_id)

        if response_type == "direct":
            result_json = {
                "text": result[0],
                "srt": result[1],
                "segments": result[2],
                "text_url": None,
                "srt_url": None,
                "segments_url": None,
            }
            return result_json, "/v1/transcribe/media/whisperx", 200

        cloud_urls = {
            "text": None,
            "srt": None,
            "segments": None,
            "text_url": upload_file(result[0]) if include_text and result[0] else None,
            "srt_url": upload_file(result[1]) if include_srt and result[1] else None,
            "segments_url": upload_file(result[2]) if include_segments and result[2] else None,
        }

        # Clean up local files if they were created
        for path in result:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        return cloud_urls, "/v1/transcribe/media/whisperx", 200

    except Exception as e:
        logger.error("Job %s (whisperx): Error during transcription process - %s", job_id, e)
        return str(e), "/v1/transcribe/media/whisperx", 500

