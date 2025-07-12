import gradio as gr
from pathlib import Path
import warnings
import os
import shutil
import json
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

warnings.filterwarnings("ignore")

from KeyFrameSelection.FeatureExtraction import process_video, save_records
from KeyFrameSelection.Similarties import hash_filter, clip_filter
from FrameProcessor.utils.io_utils import get_frames_from_folder, save_description_to_csv
from FrameProcessor.processor.multi_frame import process_frames
from config.paths import output_csv_file, output_json_file


def run_full_pipeline(video_path):
    keyframe_dir = "outputs/keyframes"
    csv_path = "outputs/keyframes.csv"

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs/final_output", exist_ok=True)

    start = time.time()

    # Step 1: Extract raw keyframes
    records, fps = process_video(video_path, interval_sec=10)

    # Step 2: Filter
    min_frames = 10
    max_iterations = 20
    iteration = 0
    hash_threshold = 5
    ssim_threshold = 0.95
    clip_threshold = 0.90
    filtered = records

    while len(filtered) >= min_frames and iteration < max_iterations:
        filtered = hash_filter(filtered, hash_threshold, ssim_threshold, 5)
        filtered = clip_filter(filtered, clip_threshold, 5)
        hash_threshold = max(1, hash_threshold - 1)
        ssim_threshold = max(0.5, ssim_threshold - 0.05)
        clip_threshold = min(0.99, clip_threshold + 0.03)
        iteration += 1

    df = save_records(filtered, keyframe_dir, csv_path, fps)

    # Step 3: Frame processing
    frame_paths = get_frames_from_folder(keyframe_dir)
    results = process_frames(frame_paths)

    important_frames = [r for r in results if r["importance"] == "important"]
    for result in important_frames:
        save_description_to_csv(result, output_csv_file)

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    end = time.time()
    return f"✅ Processed: {len(important_frames)} keyframes in {end - start:.2f}s."


def prepare_visualization_data(video_path):
    if video_path:
        return run_full_pipeline(video_path)
    else:
        raise gr.Error("A Video file is required to process.")


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style='text-align: center; color: #e91e63; line-height: 1.8; margin-bottom: 30px;'>
            <h1 style='margin-bottom: 20px;'>🎞️ Video Summarization UI</h1>
            <p style='font-size: 18px;'>Upload your lecture or tutorial video</p>
            <p style='font-size: 18px;'>Then click <b>Summarization</b> to extract key content</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            video_upload = gr.File(
                label="🎥 Upload Video",
                file_types=["video"],
                type="filepath"
            )
            btn = gr.Button("✨ Summarize", variant="primary", size="lg")
            video_name_output = gr.Textbox(label="📄 Summary Output")

    btn.click(
        fn=prepare_visualization_data,
        inputs=[video_upload],
        outputs=[video_name_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
