# ============================================================================
#  app_styles.py — stylesheet
# ============================================================================
#  Holds every visual rule in one place, so appearance can change without
#  touching interface logic.
# ============================================================================

# The whole screen fits without scrolling. The camera fills an upright
# frame, and the Start Recording control sits over the lower part of that
# same frame rather than in a separate box beneath it.
APP_CSS = """
html, body {
    height: 100% !important;
    margin: 0 !important;
    overflow: hidden !important;
    background: #12161C !important;
}

.gradio-container {
    max-width: 430px !important;
    height: 100vh !important;
    margin: 0 auto !important;
    padding: 22px 16px !important;
    background: #12161C !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}

footer { display: none !important; }

#app-title {
    text-align: center;
    color: #EDF1F5;
    font-size: 17px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0 0 16px 0;
    flex: 0 0 auto;
}

#camera-wrap {
    position: relative !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    border: 1px solid #2A313B !important;
    background: #1A1F27 !important;
}

#camera, #camera > div {
    height: 100% !important;
}

#camera video, #camera img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    border-radius: 17px !important;
}

#start {
    position: absolute !important;
    left: 50% !important;
    bottom: 26px !important;
    transform: translateX(-50%) !important;
    z-index: 30 !important;
    width: 78% !important;
    background: #3DD6C4 !important;
    color: #12161C !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 14px 0 !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4) !important;
}

#start:hover { background: #2FB5A6 !important; }
"""