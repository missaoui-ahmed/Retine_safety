"""
MedVision Streamlit Launcher
Quick start script for the testing interface
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          MedVision Testing Interface Launcher            ║
    ║                  Streamlit Edition                       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if streamlit_app.py exists
    if not Path("streamlit_app.py").exists():
        print("ERROR: streamlit_app.py not found in current directory!")
        sys.exit(1)
    
    # Check if config.yaml exists
    if not Path("config.yaml").exists():
        print("ERROR: config.yaml not found in current directory!")
        sys.exit(1)
    
    # Check if model checkpoint exists
    if not Path("checkpoints/best_model.pth").exists():
        print("WARNING: Model checkpoint not found!")
        print("         Training may be incomplete.")
    
    print("\n[1] Starting Streamlit server...")
    print("[2] Loading model and configuration...")
    print("[3] Opening browser...")
    print("\n" + "="*60)
    
    # Start streamlit
    try:
        # Run streamlit in background
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--logger.level=error"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        print("\nWaiting for server to start...")
        time.sleep(5)
        
        # Open browser
        print("Opening browser at http://localhost:8501...")
        webbrowser.open("http://localhost:8501")
        
        print("\n" + "="*60)
        print("""
        ✓ Streamlit server is running!
        
        Features available:
        ✓ 📊 Dashboard - System overview
        ✓ 🔧 System Test - Diagnostics
        ✓ 🖼️ Single Image Test - Individual predictions
        ✓ 📁 Batch Test - Multiple images
        ✓ 📈 Model Analysis - Architecture details
        ✓ ⚙️ Settings - Configuration view
        
        URL: http://localhost:8501
        
        Press Ctrl+C to stop the server.
        """)
        
        # Keep process alive
        process.wait()
    
    except KeyboardInterrupt:
        print("\n\nShutting down Streamlit server...")
        process.terminate()
        process.wait()
        print("✓ Server stopped.")
    
    except Exception as e:
        print(f"\nERROR: Failed to start Streamlit: {e}")
        print("\nTry manual start:")
        print("  streamlit run streamlit_app.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
