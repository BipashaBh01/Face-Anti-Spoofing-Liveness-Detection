# ============================================================================
#  app_interface.py — builds the Gradio interface
# ============================================================================


import gradio as gr
 
from Functions.CHL_config import APP_TITLE, START_LABEL
from Functions.CHL_app_styles import APP_CSS
 
 
def on_start():
    """Handle the Start Recording press. Screen 1 only confirms the click."""
    print("Start Recording pressed.")
    return gr.update()
 
 
def build_interface() -> gr.Blocks:
    """Build and return the Screen 1 interface."""
    with gr.Blocks(title=APP_TITLE, css=APP_CSS) as demo:
 
        gr.HTML(f'<div id="app-title">{APP_TITLE}</div>')
 
        # Camera and overlaid control share one wrapper so the button can
        # sit on top of the live feed.
        with gr.Column(elem_id="camera-wrap"):
 
            camera = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                elem_id="camera",
                show_label=False,
                container=False,
            )
 
            start = gr.Button(START_LABEL, elem_id="start")
 
        start.click(fn=on_start, outputs=[])
 
    return demo
 