# Base image
FROM python:3.10-slim

# Install system dependencies, build tools, and libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    tar \
    xz-utils \
    fonts-liberation \
    fontconfig \
    build-essential \
    yasm \
    cmake \
    meson \
    ninja-build \
    nasm \
    libssl-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libnuma-dev \
    libmp3lame-dev \
    libopus-dev \
    libvorbis-dev \
    libtheora-dev \
    libspeex-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libgnutls28-dev \
    libaom-dev \
    libdav1d-dev \
    librav1e-dev \
    libsvtav1enc-dev \
    libzimg-dev \
    libwebp-dev \
    git \
    pkg-config \
    autoconf \
    automake \
    libtool \
    libfribidi-dev \
    libharfbuzz-dev \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxrandr2 \
    libxdamage1 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install SRT from source (latest version using cmake)
RUN git clone https://github.com/Haivision/srt.git && \
    cd srt && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && rm -rf srt

# Install SVT-AV1 from source
RUN git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    cd SVT-AV1 && \
    git checkout v0.9.0 && \
    cd Build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && rm -rf SVT-AV1

# Install libvmaf from source
RUN git clone https://github.com/Netflix/vmaf.git && \
    cd vmaf/libvmaf && \
    meson build --buildtype release && \
    ninja -C build && \
    ninja -C build install && \
    cd ../.. && rm -rf vmaf && \
    ldconfig  # Update the dynamic linker cache

# Manually build and install fdk-aac (since it is not available via apt-get)
RUN git clone https://github.com/mstorsjo/fdk-aac && \
    cd fdk-aac && \
    autoreconf -fiv && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf fdk-aac

# Install libunibreak (required for ASS_FEATURE_WRAP_UNICODE)
RUN git clone https://github.com/adah1972/libunibreak.git && \
    cd libunibreak && \
    ./autogen.sh && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && rm -rf libunibreak

# Build and install libass with libunibreak support and ASS_FEATURE_WRAP_UNICODE enabled
RUN git clone https://github.com/libass/libass.git && \
    cd libass && \
    autoreconf -i && \
    ./configure --enable-libunibreak || { cat config.log; exit 1; } && \
    mkdir -p /app && echo "Config log located at: /app/config.log" && cp config.log /app/config.log && \
    make -j$(nproc) || { echo "Libass build failed"; exit 1; } && \
    make install && \
    ldconfig && \
    cd .. && rm -rf libass

# Build and install FFmpeg 8 with shared libs (libavutil.so.60 etc.) for torchcodec
RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg && \
    cd ffmpeg && \
    git checkout n8.0 && \
    PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig" \
    CFLAGS="-I/usr/include/freetype2" \
    LDFLAGS="-L/usr/lib/x86_64-linux-gnu" \
    ./configure --prefix=/usr/local \
        --enable-shared \
        --enable-gpl \
        --enable-pthreads \
        --enable-neon \
        --enable-libaom \
        --enable-libdav1d \
        --enable-librav1e \
        --enable-libsvtav1 \
        --enable-libvmaf \
        --enable-libzimg \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libwebp \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libvorbis \
        --enable-libtheora \
        --enable-libspeex \
        --enable-libass \
        --enable-libfreetype \
        --enable-libharfbuzz \
        --enable-fontconfig \
        --enable-libsrt \
        --enable-filter=drawtext \
        --extra-cflags="-I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include" \
        --extra-ldflags="-L/usr/lib/x86_64-linux-gnu -lfreetype -lfontconfig" \
        --enable-gnutls \
    && make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && rm -rf ffmpeg

# Add /usr/local/bin to PATH (if not already included)
ENV PATH="/usr/local/bin:${PATH}"

# Copy fonts into the custom fonts directory
COPY ./fonts /usr/share/fonts/custom

# Rebuild the font cache so that fontconfig can see the custom fonts
RUN fc-cache -f -v

# Set work directory
WORKDIR /app

# Set environment variable for Whisper cache
ENV WHISPER_CACHE_DIR="/app/whisper_cache"

# Create cache directory (no need for chown here yet)
RUN mkdir -p ${WHISPER_CACHE_DIR} 

# Copy the requirements file first to optimize caching
COPY requirements.txt .

# Install Python dependencies, upgrade pip
# Uninstall torchcodec: pulled in by torchaudio/pyannote but ABI-incompatible with torch 2.8+cu128 in this image.
# We do not use it (pyannote audio via soundfile+ffmpeg); removing it avoids the long torchcodec load traceback.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install openai-whisper && \
    pip install playwright && \
    pip install jsonschema && \
    pip uninstall -y torchcodec || true

# Upgrade Lightning checkpoint used by WhisperX VAD so it does not run on every process
RUN python -m lightning.pytorch.utilities.upgrade_checkpoint /usr/local/lib/python3.10/site-packages/whisperx/assets/pytorch_model.bin 2>/dev/null || true

# Pre-download Whisper base model as root (before USER switch) so cache dir is writable
RUN python -c "import os; import whisper; cache=os.environ.get('WHISPER_CACHE_DIR', '/app/whisper_cache'); whisper.load_model('base', download_root=cache)"

# Create the appuser and give ownership of /app
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Install Playwright Chromium browser as appuser
RUN playwright install chromium

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

RUN echo '#!/bin/bash\n\
gunicorn --bind 0.0.0.0:8080 \
    --workers ${GUNICORN_WORKERS:-2} \
    --timeout ${GUNICORN_TIMEOUT:-300} \
    --worker-class sync \
    --keep-alive 80 \
    --config gunicorn.conf.py \
    app:app' > /app/run_gunicorn.sh && \
    chmod +x /app/run_gunicorn.sh

# Run the shell script
CMD ["/app/run_gunicorn.sh"]
