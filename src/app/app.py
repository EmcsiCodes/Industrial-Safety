import sys
import tempfile
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference.video import process_video  # noqa: E402


st.set_page_config(
    page_title="Industrial Workplace Safety Video Detection",
    layout="centered",
)
st.title("Industrial Workplace Safety Video Detection")

uploaded_video = st.file_uploader(
    "Upload a workplace or industrial video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
)
confidence = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.80,
    value=0.30,
    step=0.05,
)
frame_interval = st.slider(
    "Detect every N frames",
    min_value=1,
    max_value=15,
    value=5,
)

process_clicked = st.button(
    "Process Video",
    type="primary",
    disabled=uploaded_video is None,
)

if process_clicked and uploaded_video is not None:
    suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
    try:
        with tempfile.TemporaryDirectory(prefix="industrial_safety_") as temporary:
            temporary_dir = Path(temporary)
            input_path = temporary_dir / f"input{suffix}"
            output_path = temporary_dir / "annotated.mp4"
            input_path.write_bytes(uploaded_video.getbuffer())

            with st.spinner("Processing video with YOLO11n..."):
                summary = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    confidence=confidence,
                    frame_interval=frame_interval,
                )
            annotated_video = output_path.read_bytes()
    except Exception as error:
        st.error(f"Video processing failed: {error}")
    else:
        frame_column, inference_column, time_column = st.columns(3)
        frame_column.metric("Total frames", summary["frames"])
        inference_column.metric("Inference frames", summary["inference_frames"])
        time_column.metric(
            "Processing time",
            f"{summary['processing_seconds']:.1f} s",
        )

        st.video(annotated_video, format="video/mp4")
        if summary["browser_compatible"]:
            st.caption("Browser-compatible H.264 MP4 output.")
        else:
            st.warning(
                "H.264 encoding is unavailable on this system, so OpenCV used "
                "mp4v. Download the video if the browser cannot play it directly."
            )
        download_name = f"{Path(uploaded_video.name).stem}_annotated.mp4"
        st.download_button(
            "Download annotated MP4",
            data=annotated_video,
            file_name=download_name,
            mime="video/mp4",
        )
