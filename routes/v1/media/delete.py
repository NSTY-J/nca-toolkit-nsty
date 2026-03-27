from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
from services.authentication import authenticate
from services.cloud_storage import delete_file
import logging

v1_media_delete_bp = Blueprint('v1_media_delete', __name__)
logger = logging.getLogger(__name__)

@v1_media_delete_bp.route('/v1/media/delete', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "media_url": {"type": "string", "format": "uri"},
    },
    "required": ["media_url"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=True)
def delete_media(job_id, data):
    media_url = data['media_url']
    # Normalize internal hostnames (host.docker.internal -> minio)
    media_url = media_url.replace('host.docker.internal', 'minio')
    object_key = delete_file(media_url)
    return {"deleted": object_key}, "/v1/media/delete", 200
