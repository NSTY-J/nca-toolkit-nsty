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
# 51 Franklin Street, Fifth Floor, Boston, MA 02100-1301 USA.

import os
import uuid
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from services.cloud_storage import upload_file
from services.authentication import authenticate
from config import LOCAL_STORAGE_PATH

v1_media_upload_bp = Blueprint('v1_media_upload', __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    'mp4', 'webm', 'mkv', 'avi', 'mov', 'mp3', 'wav', 'm4a', 'ogg', 'flac'
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@v1_media_upload_bp.route('/v1/media/upload', methods=['POST'])
@authenticate
def upload_media():
    """
    Accept multipart file upload, save to temp, upload to cloud storage, return media_url.
    Used for transcribe UI - upload video/audio before triggering n8n workflow.
    """
    if 'file' not in request.files:
        return jsonify({"message": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "message": f"File type not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    temp_path = None
    try:
        safe_name = secure_filename(file.filename) or 'upload'
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        temp_path = os.path.join(LOCAL_STORAGE_PATH, unique_name)

        file.save(temp_path)
        logger.info(f"Saved upload to {temp_path}")

        media_url = upload_file(temp_path)
        os.remove(temp_path)
        logger.info(f"Uploaded to cloud: {media_url}")

        return jsonify({"media_url": media_url}), 200
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return jsonify({"message": str(e)}), 500
