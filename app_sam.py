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
from yt_dlp import YoutubeDL

# Initialize session state for selected indices
if 'selected_indices' not in st.session_state:
    st.session_state.selected_indices = []

load_dotenv()
token = os.getenv('HUGGING_FACE_HUB_TOKEN')
login(token=token)

def build_base_network(input_shape):
    """Create base CNN for feature extraction"""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return x

def get_youtube_stream_url(video_id):
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
        return info['url']

def extract_frames_from_stream(video_url, interval=1):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise Exception("Failed to open video stream")
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_indices = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (int(frame_rate) * interval) == 0:
            frames.append(frame)
            frame_indices.append(frame_count)
        frame_count += 1
    
    cap.release()
    return frames, frame_indices, frame_rate, total_frames

def get_context_frames(video_url, frame_index, context_seconds=5, fps=30):
    context_frames = []
    cap = cv2.VideoCapture(video_url)
    
    if not cap.isOpened():
        raise Exception("Failed to open video stream")
    
    # Calculate start and end frames
    start_frame = max(0, frame_index - int(fps * context_seconds))
    end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 
                   frame_index + int(fps * context_seconds))
    
    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read frames
    frame_count = start_frame
    while cap.isOpened() and frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        context_frames.append(frame)
        frame_count += 1
    
    cap.release()
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
    
    height, width, layers = frames[0].shape
    
    # Define a list of codecs to try, in order of preference
    codec_options = [
        ('XVID', '.avi'),  # XVID codec with AVI container
        ('MJPG', '.avi'),  # Motion JPEG with AVI container
        ('mp4v', '.mp4'),  # MPEG-4 with MP4 container
        ('X264', '.mp4'),  # H.264 with MP4 container
        ('DIV3', '.avi'),  # DivX3 with AVI container
    ]
    
    last_error = None
    for codec, extension in codec_options:
        try:
            # Update output path with correct extension
            current_output = os.path.splitext(output_path)[0] + extension
            
            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(current_output, fourcc, fps, (width, height))
            
            if out.isOpened():
                for frame in frames:
                    out.write(frame)
                out.release()
                
                # If we successfully wrote the video, read it back to verify
                with open(current_output, 'rb') as f:
                    video_bytes = f.read()
                    if len(video_bytes) > 0:
                        return current_output, video_bytes
        except Exception as e:
            last_error = str(e)
            continue
        finally:
            if 'out' in locals():
                out.release()
    
    raise Exception(f"Failed to create video with any available codec. Last error: {last_error}")

try:
    input_shape = (224, 224, 3)
    
    # Create inputs
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # Share base network weights for both inputs
    tower = tf.keras.models.Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu')
    ])
    
    # Get embeddings
    embedding_a = tower(input_a)
    embedding_b = tower(input_b)
    
    # Compute absolute difference between embeddings
    distance = Lambda(lambda x: tf.math.abs(x[0] - x[1]))([embedding_a, embedding_b])
    
    # Add prediction layer
    output = Dense(1, activation='sigmoid')(distance)
    
    # Create model
    siamese_model = Model(inputs=[input_a, input_b], outputs=output)
    
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

    # Move YouTube ID input to sidebar
    with st.sidebar:
        st.header("Input Video")
        video_id = st.text_input(
            "Enter YouTube Video ID",
            help="Found in the URL after 'v='. Video must be under 10 minutes."
        )

    # Create two columns for layout
    col1, col2 = st.columns(2)

    frames = []
    stream_url = None

    if video_id:
        with st.spinner("Processing YouTube video..."):
            stream_url = get_youtube_stream_url(video_id)
            frames, frame_indices, frame_rate, total_frames = extract_frames_from_stream(stream_url)
            st.sidebar.success(f"Extracted {len(frames)} frames from the video.")

    with col1:
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
                        temp_dir = tempfile.mkdtemp()
                        temp_file = os.path.join(temp_dir, 'output.mp4')
                        
                        # Get context frames for each selected frame
                        all_context_frames = []
                        for frame_idx in st.session_state.selected_indices:
                            context_frames = get_context_frames(stream_url, frame_indices[frame_idx], 
                                                             context_seconds=5, fps=frame_rate)
                            all_context_frames.extend(context_frames)
                        
                        if all_context_frames:
                            # Create video file with new function
                            output_path, video_bytes = create_video_from_frames(all_context_frames, temp_file, fps=frame_rate)
                            
                            if len(video_bytes) > 0:
                                st.video(video_bytes)
                                # Also provide a download button
                                st.download_button(
                                    label="Download Video",
                                    data=video_bytes,
                                    file_name=os.path.basename(output_path),
                                    mime="video/mp4" if output_path.endswith('.mp4') else "video/avi"
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
                            for ext in ['.mp4', '.avi']:
                                possible_file = os.path.splitext(temp_file)[0] + ext
                                if os.path.exists(possible_file):
                                    os.remove(possible_file)
                            os.rmdir(temp_dir)
                        except Exception as e:
                            st.warning(f"Failed to clean up temporary files: {str(e)}")
        else:
            st.info("No frames selected yet. Please select frames from the left column.")

except Exception as e:
    st.error(f"Error loading model: {str(e)}")
