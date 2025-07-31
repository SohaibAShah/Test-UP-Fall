# config.py

# --- Dataset Paths ---
# Base directory for the UP-Fall Dataset
# Assumes 'UP-Fall Dataset' is in the same directory as your Python scripts
DATASET_BASE_DIR = 'UP-Fall Dataset'
SENSOR_DATA_PATH = os.path.join(DATASET_BASE_DIR, 'Imp_sensor.csv')
DOWNLOADED_CAMERA_FILES_DIR = os.path.join(DATASET_BASE_DIR, 'downloaded_camera_files')
# Temporary directory for extracting zip files
TEMP_CAMERA_EXTRACTION_DIR = 'CAMERA_TEMP'

# --- Image Processing Parameters ---
DEFAULT_IMAGE_WIDTH = 32
DEFAULT_IMAGE_HEIGHT = 32

# --- Subject, Activity, Camera Ranges ---
START_SUBJECT = 1
END_SUBJECT = 17
START_ACTIVITY = 1
END_ACTIVITY = 11
START_CAMERA = 1
END_CAMERA = 2 # Changed to include both camera 1 and 2 in a single run if desired

# --- Output File Names ---
IMAGE_FILE_PREFIX = 'image_'
NAME_FILE_PREFIX = 'name_'
LABEL_FILE_PREFIX = 'label_'
NPY_EXTENSION = '.npy'

# --- Specific Exclusions/Problematic Files ---
# Define problematic trials/cameras to skip during loading
# Format: (subject_id, activity_id, trial_id, camera_id)
# Based on the original notebook's explicit skips
# (sub_ == 8 and act_ == 11) and ( trial_ == 2 or trial_ == 3) for Camera1
# (sub_ == 6 and act_ == 10) and ( trial_ == 2 ) for Camera2
# The current load_img handles Camera1 specific exclusions.
# For Camera2, the original code had a specific skip for Subject6Activity10Trial2Camera2,
# but it was within the same 'if' block as the length check, so a more robust check might be needed.
# For now, we'll rely on the load_img's internal checks.

# Note on 'Invalid image' and 'NO SHAPE' issues:
# The original notebook includes print statements like 'Invalid image' and 'NO SHAPE'
# based on `len(filepath) > 70` and `filepath == 'CAMERA/Subject6Activity10Trial2Camera2/2018-07-06T12_03_04.483526.png'`.
# These specific hardcoded paths and length checks are kept within load_img for direct translation.
# In a real-world scenario, you might want more dynamic or robust error handling for image loading.