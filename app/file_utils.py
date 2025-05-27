import os
import zipfile
import tempfile
from pathlib import Path

def find_audio_file(segment_row, audio_dir="./audio/segments/"):
    if not os.path.exists(audio_dir):
        return None
        
    #files_map = {f.lower(): f for f in os.listdir(audio_dir)}
    
    possible_names = [
        segment_row['filename'],
        segment_row['filename'].lower(),
    ]
    
    if 'audio_filename' in segment_row and 'device_id' in segment_row and 'start_time' in segment_row:
        base_filename = os.path.splitext(segment_row['audio_filename'])[0]
        segment_start = int(segment_row['start_time'])
        index = int(segment_start / 3)
        transformed = f"{base_filename}_{segment_row['device_id']}_{index}.wav"
        possible_names.append(transformed)
        possible_names.append(transformed.lower())
    
    for name in possible_names:
        path = os.path.join(audio_dir, name)
        if os.path.exists(path):
            return path

        # Case-insensitive match
        #if name.lower() in files_map:
        #return os.path.join(audio_dir, files_map[name.lower()])
            
    return None

def find_embedding_file(segment_row, embedding_dir="./audio/embeddings/"):
    if not os.path.exists(embedding_dir):
        return None
        
    # Get all files in directory with case-insensitive lookup
    #files_map = {f.lower(): f for f in os.listdir(embedding_dir)}
    
    # Try different filename formats
    possible_names = []
    
    # Format 1: Basic approach - just change extension
    base_name = os.path.splitext(segment_row['filename'])[0].lower()
    possible_names.append(f"{base_name}.pt")
    
    # Format 2: Try with device ID and index
    if 'audio_filename' in segment_row and 'device_id' in segment_row and 'start_time' in segment_row:
        base_filename = os.path.splitext(segment_row['audio_filename'])[0]
        segment_start = int(segment_row['start_time'])
        index = int(segment_start / 3)
        transformed = f"{base_filename.lower()}_{segment_row['device_id']}_{index}.pt"
        possible_names.append(transformed)
    
    # Try each name
    for name in possible_names:
        # Direct match
        path = os.path.join(embedding_dir, name)
        if os.path.exists(path):
            return path
            
        # Case-insensitive match
        #if name.lower() in files_map:
        #    return os.path.join(embedding_dir, files_map[name.lower()])
            
    return None

def create_zip_archive(files, prefix="export"):
    if not files:
        return None
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        zip_path = tmp_zip.name
        
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            file_path = Path(file)
            if file_path.exists():
                zipf.write(file_path, arcname=file_path.name)
                
    return zip_path