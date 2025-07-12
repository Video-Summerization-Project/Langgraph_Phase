import gradio as gr
from pathlib import Path

def prepare_visualization_data(video_path):
    if video_path:
        print(f"✅ Video background provided: {video_path}")
    else:
        raise gr.Error("A Video file is required to generate a visualization.")
    return f"🎥 {Path(video_path).name} ✅ Done"

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
            video_name_output = gr.Textbox(label="📄 Video Summary Output")

    btn.click(
        fn=prepare_visualization_data,
        inputs=[video_upload],
        outputs=[video_name_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
