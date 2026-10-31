# ============================================================================
#  main_fas.py — application entry point
# ============================================================================


from Functions.CHL_server import run_server
 
 
def main() -> None:
    """Launch the web server."""
    run_server(host="127.0.0.1", port=8000, open_browser=True)
 
 
if __name__ == "__main__":
    main()