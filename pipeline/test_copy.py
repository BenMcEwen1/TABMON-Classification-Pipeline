import shutil

src = "audio/test_bugg/2024-05-07T12_00_57.834Z.mp3"
dest = "audio/tests/2024-05-07T12_00_57.834Z.mp3"

try:
    shutil.copy2(src, dest)
    print(f"Successfully copied {src} to {dest}")
except Exception as e:
    print(f"Error copying {src} to {dest}: {e}")
