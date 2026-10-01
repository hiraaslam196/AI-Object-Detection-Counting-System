import streamlit as st
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Page settings
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("AI Object Detection & Counting System")
st.write("Detect and count objects in images and videos using AI.")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Select input type
input_type = st.radio(
    "Choose Input Type",
    ["Image", "Video"]
)

# =========================================================
# IMAGE DETECTION
# =========================================================

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        if st.button("Detect Objects"):

            image_array = np.array(image)

            # Run YOLO
            results = model(image_array)

            # Count objects
            object_counts = Counter()

            for result in results:

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    object_counts[class_name] += 1

            # Draw bounding boxes
            annotated_image = results[0].plot()

            st.subheader("Detection Result")
            st.image(
                annotated_image,
                use_container_width=True
            )

            # Display counts
            st.subheader("Detected Objects")

            if object_counts:

                total_objects = sum(object_counts.values())

                st.write(
                    f"### Total Objects: {total_objects}"
                )

                for object_name, count in object_counts.items():

                    st.write(
                        f"**{object_name.capitalize()}:** {count}"
                    )

            else:

                st.warning("No objects detected.")


# =========================================================
# VIDEO DETECTION
# =========================================================

else:

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(uploaded_video.read())
        temp_input.close()

        st.subheader("Original Video")

        st.video(uploaded_video)

        if st.button("Detect Objects in Video"):

            st.info("Processing video... Please wait.")

            # Open video
            cap = cv2.VideoCapture(temp_input.name)

            # Video information
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            # Temporary output video
            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            # Overall object counter
            object_counts = Counter()

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # Run YOLO
                results = model(frame)

                # Count detected objects
                for result in results:

                    for box in result.boxes:

                        class_id = int(box.cls[0])

                        class_name = model.names[class_id]

                        object_counts[class_name] += 1

                    # Draw boxes
                    annotated_frame = result.plot()

                    out.write(annotated_frame)

            # Release video
            cap.release()
            out.release()

            st.success("Video processing completed!")

            # Display processed video
            st.subheader("Detection Result")

            with open(output_path, "rb") as video_file:

                video_bytes = video_file.read()

            st.video(video_bytes)

            # Display counts
            st.subheader("Detected Objects")

            if object_counts:

                total_objects = sum(
                    object_counts.values()
                )

                st.write(
                    f"### Total Detected Objects: {total_objects}"
                )

                for object_name, count in object_counts.items():

                    st.write(
                        f"**{object_name.capitalize()}: {count}**"
                    )

            else:

                st.warning(
                    "No objects detected in the video."
                )

            # Remove temporary files
            os.remove(temp_input.name)
            os.remove(output_path)