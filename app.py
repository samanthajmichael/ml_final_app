# app.py
import streamlit as st
import cv2
import os
import re
import shutil
import tempfile
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub import login
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model, Sequential
from yt_dlp import YoutubeDL

# Constants
MAX_VIDEO_LENGTH_MINUTES = 10
FRAME_BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
MODEL_REPO_ID = "samanthajmichael/siamese_model.h5"
MODEL_FILENAME = "siamese_model.h5"

# Initialize session state
def init_session_state():
    if 'selected_indices' not in st.session_state:
        st.session_state.selected_indices = []
    if 'processed_video_id' not in st.session_state:
        st.session_state.processed_video_id = None

@contextmanager
def video_capture(url: str):
    """Context manager for video capture."""
    cap = None
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise ValueError("Failed to open video stream")
        yield cap
    finally:
        if cap is not None:
            cap.release()

def validate_youtube_id(video_id: str) -> bool:
    """Validate YouTube video ID format."""
    if not video_id:
        return False
    return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))

@st.cache_resource
def load_siamese_model():
    """Load and cache the Siamese model."""
    try:
        # Create base tower
        tower = Sequential([
            Input(shape=INPUT_SHAPE),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu')
        ])

        # Create inputs
        input_a = Input(shape=INPUT_SHAPE)
        input_b = Input(shape=INPUT_SHAPE)

        # Get embeddings
        embedding_a = tower(input_a)
        embedding_b = tower(input_b)

        # Compute distance
        distance = Lambda(lambda x: tf.math.abs(x[0] - x[1]))([embedding_a, embedding_b])
        output = Dense(1, activation='sigmoid')(distance)

        # Create and compile model
        model = Model(inputs=[input_a, input_b], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Load weights
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        raise

def get_youtube_stream_url(video_id: str) -> str:
    """Get YouTube video stream URL with error handling."""
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'max_filesize': 1024 * 1024 * 100  # 100MB limit
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
            duration = info.get('duration', 0)
            if duration > MAX_VIDEO_LENGTH_MINUTES * 60:
                raise ValueError(f"Video exceeds maximum length of {MAX_VIDEO_LENGTH_MINUTES} minutes")
            return info['url']
    except Exception as e:
        raise ValueError(f"Failed to process YouTube video: {str(e)}")

def extract_frames_from_stream(video_url: str, interval: int = 1) -> Tuple[List[np.ndarray], List[int], float, int]:
    """Extract frames from video stream with progress bar."""
    frames = []
    frame_indices = []
    
    with video_capture(video_url) as cap:
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(frame_rate * interval)
        
        if frame_interval < 1:
            frame_interval = 1

        progress_bar = st.progress(0)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
                frame_indices.append(frame_count)
                
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    return frames, frame_indices, frame_rate, total_frames

def find_similar_frames_batch(reference_image: np.ndarray, 
                            candidate_frames: List[np.ndarray], 
                            model: Model,
                            top_n: int = 5) -> List[Tuple[int, float]]:
    """Find similar frames using batch processing."""
    reference_image = cv2.resize(reference_image, (224, 224)) / 255.0
    reference_image = reference_image[np.newaxis, ...]
    
    similarities = []
    progress_bar = st.progress(0)
    
    for i in range(0, len(candidate_frames), FRAME_BATCH_SIZE):
        batch = candidate_frames[i:min(i + FRAME_BATCH_SIZE, len(candidate_frames))]
        batch = [cv2.resize(frame, (224, 224)) / 255.0 for frame in batch]
        batch = np.stack(batch)
        
        # Repeat reference image for batch processing
        reference_batch = np.repeat(reference_image, len(batch), axis=0)
        
        similarity_batch = model.predict([reference_batch, batch], verbose=0)
        similarities.extend([(i + j, sim[0]) for j, sim in enumerate(similarity_batch)])
        
        progress_bar.progress(min((i + FRAME_BATCH_SIZE) / len(candidate_frames), 1.0))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

def create_video_clip(frames: List[np.ndarray], fps: float = 30) -> Optional[bytes]:
    """Create video from frames with multiple codec attempts."""
    if not frames:
        raise ValueError("No frames provided for video creation")

    height, width, _ = frames[0].shape
    temp_dir = tempfile.mkdtemp()
    
    try:
        codecs = [
            ('XVID', '.avi'),
            ('MJPG', '.avi'),
            ('mp4v', '.mp4'),
            ('X264', '.mp4')
        ]

        for codec, extension in codecs:
            try:
                output_path = os.path.join(temp_dir, f'output{extension}')
                fourcc = cv2.VideoWriter_fourcc(*codec)
                
                with st.spinner(f"Attempting to create video with {codec} codec..."):
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    if out.isOpened():
                        for frame in frames:
                            out.write(frame)
                        out.release()
                        
                        # Verify the video file
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            with open(output_path, 'rb') as f:
                                return f.read()
            except Exception as e:
                st.warning(f"Failed with codec {codec}: {str(e)}")
                continue
            finally:
                if 'out' in locals():
                    out.release()
        
        raise ValueError("Failed to create video with any available codec")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    st.set_page_config(page_title="Match Cutting with YouTube", layout="wide")
    init_session_state()
    
    # Load environment variables and authenticate
    load_dotenv()
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not token:
        st.error("Hugging Face token not found in environment variables")
        return
    
    try:
        login(token=token)
    except Exception as e:
        st.error(f"Failed to authenticate with Hugging Face: {str(e)}")
        return

    # Load model
    try:
        model = load_siamese_model()
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")
        return

    st.title("Match Cutting with YouTube")

    # Sidebar inputs
    with st.sidebar:
        st.header("Input Video")
        video_id = st.text_input(
            "Enter YouTube Video ID",
            help="Found in the URL after 'v='. Example: dQw4w9WgXcQ"
        )

    # Main content
    col1, col2 = st.columns(2)

    # Process video
    frames = []
    if video_id and validate_youtube_id(video_id):
        if video_id != st.session_state.processed_video_id:
            try:
                with st.spinner("Processing YouTube video..."):
                    stream_url = get_youtube_stream_url(video_id)
                    frames, frame_indices, frame_rate, total_frames = extract_frames_from_stream(stream_url)
                    st.session_state.processed_video_id = video_id
                    st.session_state.frames = frames
                    st.session_state.frame_indices = frame_indices
                    st.session_state.frame_rate = frame_rate
            except Exception as e:
                st.error(str(e))
                return
        else:
            frames = st.session_state.frames
            frame_indices = st.session_state.frame_indices
            frame_rate = st.session_state.frame_rate

    # Frame selection and matching
    with col1:
        if frames:
            st.write("Select a frame to use as the reference:")
            selected_frame_index = st.slider("Frame index:", 0, len(frames) - 1, 0)
            selected_frame = frames[selected_frame_index]
            st.image(selected_frame, caption="Reference Frame", use_column_width=True)

            with st.spinner("Finding similar frames..."):
                top_matches = find_similar_frames_batch(selected_frame, frames, model)

            st.write("Similar Frames:")
            st.session_state.selected_indices = []
            
            for rank, (frame_idx, similarity) in enumerate(top_matches, 1):
                col_img, col_check = st.columns([3, 1])
                with col_img:
                    st.image(frames[frame_idx], 
                            caption=f"Match {rank} (Score: {similarity:.2f})", 
                            use_column_width=True)
                with col_check:
                    if st.checkbox(f"Select {rank}", key=f"check_{frame_idx}"):
                        st.session_state.selected_indices.append(frame_idx)

    # Video creation
    with col2:
        if st.session_state.selected_indices:
            st.write("Selected Frames:")
            for idx, frame_idx in enumerate(st.session_state.selected_indices):
                st.image(frames[frame_idx], 
                        caption=f"Selected Frame {idx + 1}", 
                        use_column_width=True)

            if st.button("Create Video"):
                try:
                    selected_frames = [frames[idx] for idx in st.session_state.selected_indices]
                    video_bytes = create_video_clip(selected_frames, fps=frame_rate)
                    
                    if video_bytes:
                        st.video(video_bytes)
                        st.download_button(
                            "Download Video",
                            video_bytes,
                            "output.mp4",
                            "video/mp4"
                        )
                except Exception as e:
                    st.error(f"Failed to create video: {str(e)}")
        else:
            st.info("Select frames from the left column to create a video.")

if __name__ == "__main__":
    main()
