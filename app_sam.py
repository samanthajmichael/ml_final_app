import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub import login
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from yt_dlp import YoutubeDL

# Initialize session state for selected indices
if 'selected_indices' not in st.session_state:
    st.session_state.selected_indices = []

load_dotenv()
token = os.getenv('HUGGING_FACE_HUB_TOKEN')
login(token=token)

def build_base_network(input_shape):
    """Create base CNN for feature extraction"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return x

def get_youtube_stream_url(video_id):
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
        return info['url']

def extract_frames_from_stream(video_url, interval=1):
    clip = VideoFileClip(video_url)
    frames = [frame for frame in clip.iter_frames(fps=clip.fps)]
    frame_indices = [i for i in range(0, len(frames), int(clip.fps * interval))]
    return frames, frame_indices, clip.fps, len(frames)

def get_context_frames(video_url, frame_index, context_seconds=5, fps=30):
    clip = VideoFileClip(video_url)
    start_time = max(0, frame_index / fps - context_seconds)
    end_time = min(clip.duration, frame_index / fps + context_seconds)
    context_clip = clip.subclip(start_time, end_time)
    context_frames = [frame for frame in context_clip.iter_frames()]
    return context_frames

def find_similar_frames(reference_image, candidate_frames, top_n=5):
    reference_image = cv2.resize(reference_image, (224, 224)) / 255.0
    reference_image = reference_image[np.newaxis, ...]
    
    similarities = []
    for i, frame in enumerate(candidate_frames):
        candidate_image = cv2.resize(frame, (224, 224)) / 255.0
        candidate_image = candidate_image[np.newaxis, ...]
        similarity = siamese_model.predict([reference_image, candidate_image], verbose=0)[0][0]
        similarities.append((i, similarity))
    
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def create_video_from_frames(frames, output_path, fps=30):
    if not frames:
        raise ValueError("No frames provided for video creation")
    
    clips = [VideoFileClip(f'memory://{i}').set_duration(1/fps) for i, frame in enumerate(frames)]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, fps=fps)

try:
    input_shape = (224, 224, 3)
    
    # Create inputs
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)
    
    # Share base network weights for both inputs
    tower = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu')
    ])
    
    # Get embeddings
    embedding_a = tower(input_a)
    embedding_b = tower(input_b)
    
    # Compute absolute difference between embeddings
    distance = tf.keras.layers.Lambda(lambda x: tf.math.abs(x[0] - x[1]))([embedding_a, embedding_b])
    
    # Add prediction layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    
    # Create model
    siamese_model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=output)
    
    # Compile model
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Download and load weights
    model_path = hf_hub_download(
        repo_id="samanthajmichael/siamese_model.h5",
        filename="siamese_model.h5"
    )
    
    # Load weights
    siamese_model.load_weights(model_path)
    st.success("Model loaded successfully!")

    # Streamlit app
    st.title("Match Cutting with YouTube")

    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        # Input for YouTube Video ID
        video_id = st.text_input("Enter YouTube Video ID (Found in the URL (after 'v=') Video must be under 10 minutes):")
        frames = []
        stream_url = None

        if video_id:
            with st.spinner("Processing YouTube video..."):
                stream_url = get_youtube_stream_url(video_id)
                frames, frame_indices, frame_rate, total_frames = extract_frames_from_stream(stream_url)
                st.write(f"Extracted {len(frames)} frames from the video.")

        # Select and display a frame
        if frames:
            st.write("Select a frame to use as the reference:")
            selected_frame_index = st.slider("Select a frame index:", min_value=0, max_value=len(frames) - 1, value=0)
            selected_frame = frames[selected_frame_index]
            st.image(selected_frame, caption="Selected Reference Frame", use_container_width=True)
            
            # Perform similarity analysis
            with st.spinner("Finding similar frames..."):
                top_matches = find_similar_frames(selected_frame, frames, top_n=5)
            
            # Display similar frames with checkboxes
            st.write("Top Similar Frames:")
            st.session_state.selected_indices = []  # Reset at start of selection
            for rank, (frame_idx, similarity) in enumerate(top_matches, 1):
                col_img, col_check = st.columns([3, 2])
                with col_img:
                    st.image(frames[frame_idx], caption=f"Match {rank} - Similarity Score: {similarity:.2f}", use_container_width=True)
                with col_check:
                    if st.checkbox(f"Select Frame {rank}", key=f"check_{frame_idx}"):
                        st.session_state.selected_indices.append(frame_idx)

    with col2:
        if len(st.session_state.selected_indices) > 0:
            st.write("Selected Frames:")
            
            # Display selected frames first
            for idx, frame_idx in enumerate(st.session_state.selected_indices):
                st.image(frames[frame_idx], caption=f"Selected Frame {idx + 1}", use_container_width=True)
            
            # Create a temporary video file from selected frames
            if st.button("Create Video from Selected Frames"):
                with st.spinner("Creating video..."):
                    try:
                        # Create temporary directory with a specific name
                        temp_dir = tempfile.mkdtemp()
                        temp_file = os.path.join(temp_dir, 'output.mp4')
                        
                        # Get context frames for each selected frame
                        all_context_frames = []
                        for frame_idx in st.session_state.selected_indices:
                            context_frames = get_context_frames(stream_url, frame_indices[frame_idx], 
                                                             context_seconds=5, fps=frame_rate)
                            all_context_frames.extend(context_frames)
                        
                        if all_context_frames:
                            # Create video file
                            create_video_from_frames(all_context_frames, temp_file, fps=frame_rate)
                            
                            # Read and display the video
                            with open(temp_file, 'rb') as video_file:
                                video_bytes = video_file.read()
                            
                            if len(video_bytes) > 0:
                                st.video(video_bytes)
                                # Also provide a download button
                                st.download_button(
                                    label="Download Video",
                                    data=video_bytes,
                                    file_name="match_cut.mp4",
                                    mime="video/mp4"
                                )
                            else:
                                st.error("Generated video file is empty")
                        else:
                            st.error("No frames were collected for the video")
                            
                    except Exception as e:
                        st.error(f"Error in video processing: {str(e)}")
                    finally:
                        # Clean up
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                            os.rmdir(temp_dir)
                        except Exception as e:
                            st.warning(f"Failed to clean up temporary files: {str(e)}")
        else:
            st.info("No frames selected yet. Please select frames from the left column.")

except Exception as e:
    st.error(f"Error loading model: {str(e)}")
