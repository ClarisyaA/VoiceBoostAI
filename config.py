import os
from pathlib import Path

# Application Settings
APP_NAME = "VoiceBoost"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-Powered Dysarthria Classification System"

# Paths Configuration
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_TRAINING_DIR = BASE_DIR / "data_training"
DATA_TESTING_DIR = BASE_DIR / "data_testing"
RECORDINGS_DIR = BASE_DIR / "recordings"
TEMP_DIR = BASE_DIR / "temp"

# Model Configuration
MODEL_FILE = MODELS_DIR / "dysarthria_classifier.pkl"
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# Audio Configuration
AUDIO_CONFIG = {
    'sample_rate': 22050,
    'channels': 1,
    'chunk_size': 1024,
    'format': 'wav',
    'max_duration': 30,  # seconds
    'min_duration': 1,   # seconds
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    'mfcc_coefficients': 13,
    'formant_max_frequency': 5500,
    'formant_time_step': 0.025,
    'formant_max_formants': 5,
    'pitch_min_frequency': 75,
    'pitch_max_frequency': 600,
}

# Text-to-Speech Configuration
TTS_CONFIG = {
    'language': 'en',
    'slow': False,
    'lang_options': {
        'English': 'en',
        'Indonesian': 'id',
        'Spanish': 'es',
        'French': 'fr',
        'German': 'de',
    }
}

# Practice Text Configuration
PRACTICE_WORDS = [
    # Basic words
    "hello", "world", "computer", "artificial", "intelligence",
    "machine", "learning", "python", "programming", "technology",
    
    # Challenging pronunciation
    "beautiful", "wonderful", "amazing", "fantastic", "incredible",
    "elephant", "butterfly", "rainbow", "sunshine", "mountain",
    
    # Medical/Speech related
    "speech", "therapy", "communication", "articulation", "pronunciation",
    "diagnosis", "treatment", "rehabilitation", "assessment", "evaluation",
    
    # Common daily words
    "family", "friend", "house", "water", "food",
    "telephone", "television", "newspaper", "hospital", "doctor"
]

PRACTICE_SENTENCES = [
    # Pangrams and tongue twisters
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    
    # Technology related
    "Artificial intelligence is changing the world rapidly.",
    "Python is a powerful programming language for data science.",
    "Speech recognition technology is advancing every day.",
    "Machine learning helps us solve complex problems efficiently.",
    
    # Daily conversation
    "Good morning, how are you feeling today?",
    "The weather is beautiful and sunny outside.",
    "I love listening to music in the morning.",
    "Technology makes our lives easier and better.",
    "Reading books expands our knowledge and imagination.",
    "Communication is the key to understanding each other.",
    
    # Medical/Speech therapy
    "Please speak clearly and slowly for the recording.",
    "Regular practice improves speech clarity significantly.",
    "The speech therapist recommended daily exercises.",
    "Clear articulation is important for effective communication.",
    
    # Challenging sentences
    "She sells seashells by the seashore.",
    "Red leather, yellow leather, red leather, yellow leather.",
    "Unique New York, unique New York, unique New York.",
    "Toy boat, toy boat, toy boat, toy boat."
]

# UI Configuration
UI_CONFIG = {
    'page_icon': '🎤',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'theme': {
        'primary_color': '#FF6B6B',
        'background_color': '#FFFFFF',
        'secondary_background_color': '#F0F2F6',
        'text_color': '#262730',
    }
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'confidence_threshold': 0.7,
    'quality_thresholds': {
        'min_snr': 15,  # dB
        'min_duration': 1,  # seconds
        'max_duration': 30,  # seconds
        'min_rms': 0.01,
        'max_peak': 0.99,
    },
    'classification_labels': {
        0: 'Normal',
        1: 'Dysarthria'
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': BASE_DIR / 'app.log',
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories"""
    directories = [
        MODELS_DIR,
        DATA_TRAINING_DIR,
        DATA_TESTING_DIR,
        RECORDINGS_DIR,
        TEMP_DIR,
        DATA_TRAINING_DIR / "dysarthria",
        DATA_TRAINING_DIR / "normal",
        DATA_TESTING_DIR / "dysarthria",
        DATA_TESTING_DIR / "normal"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Environment Variables
def get_env_config():
    """Get configuration from environment variables"""
    return {
        'DEBUG': os.getenv('DEBUG', 'False').lower() == 'true',
        'PORT': int(os.getenv('PORT', 8501)),
        'HOST': os.getenv('HOST', 'localhost'),
        'MODEL_PATH': os.getenv('MODEL_PATH', str(MODEL_FILE)),
        'DATA_PATH': os.getenv('DATA_PATH', str(DATA_TRAINING_DIR)),
    }

# Validation functions
def validate_audio_file(file_path):
    """Validate audio file format and quality"""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file extension
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        return False, "Invalid file format"
    
    try:
        import librosa
        # Try to load the file
        y, sr = librosa.load(file_path, sr=None)
        
        # Check duration
        duration = len(y) / sr
        if duration < AUDIO_CONFIG['min_duration']:
            return False, f"Audio too short (minimum {AUDIO_CONFIG['min_duration']}s)"
        
        if duration > AUDIO_CONFIG['max_duration']:
            return False, f"Audio too long (maximum {AUDIO_CONFIG['max_duration']}s)"
        
        return True, "Valid audio file"
        
    except Exception as e:
        return False, f"Error loading audio: {str(e)}"

def validate_model_file():
    """Validate model file exists and is loadable"""
    if not MODEL_FILE.exists():
        return False, "Model file not found"
    
    try:
        import pickle
        with open(MODEL_FILE, 'rb') as f:
            model_data = pickle.load(f)
        
        required_keys = ['model', 'scaler']
        for key in required_keys:
            if key not in model_data:
                return False, f"Model file missing '{key}'"
        
        return True, "Model file is valid"
        
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

# Helper functions
def get_random_practice_text(text_type='word'):
    """Get random practice text"""
    import random
    
    if text_type.lower() == 'word':
        return random.choice(PRACTICE_WORDS)
    elif text_type.lower() == 'sentence':
        return random.choice(PRACTICE_SENTENCES)
    else:
        return random.choice(PRACTICE_WORDS + PRACTICE_SENTENCES)

def get_supported_languages():
    """Get supported TTS languages"""
    return TTS_CONFIG['lang_options']

def get_app_info():
    """Get application information"""
    return {
        'name': APP_NAME,
        'version': APP_VERSION,
        'description': APP_DESCRIPTION,
    }

# Configuration validation
def validate_config():
    """Validate application configuration"""
    errors = []
    
    # Check if required directories exist
    if not MODELS_DIR.exists():
        errors.append(f"Models directory not found: {MODELS_DIR}")
    
    # Check model file
    model_valid, model_msg = validate_model_file()
    if not model_valid:
        errors.append(f"Model validation failed: {model_msg}")
    
    # Check if required packages are available
    required_packages = [
        'streamlit', 'librosa', 'parselmouth', 
        'whisper', 'gtts', 'scikit-learn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            errors.append(f"Required package not found: {package}")
    
    return len(errors) == 0, errors

# Initialize configuration
def initialize_app():
    """Initialize application configuration"""
    # Create directories
    create_directories()
    
    # Validate configuration
    is_valid, errors = validate_config()
    
    if not is_valid:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print(f"{APP_NAME} v{APP_VERSION} initialized successfully!")
    return True

# Export configuration
__all__ = [
    'APP_NAME', 'APP_VERSION', 'APP_DESCRIPTION',
    'MODELS_DIR', 'DATA_TRAINING_DIR', 'DATA_TESTING_DIR', 'RECORDINGS_DIR',
    'MODEL_FILE', 'WHISPER_MODEL',
    'AUDIO_CONFIG', 'FEATURE_CONFIG', 'TTS_CONFIG',
    'PRACTICE_WORDS', 'PRACTICE_SENTENCES',
    'UI_CONFIG', 'ANALYSIS_CONFIG', 'LOGGING_CONFIG',
    'create_directories', 'get_env_config',
    'validate_audio_file', 'validate_model_file',
    'get_random_practice_text', 'get_supported_languages',
    'get_app_info', 'validate_config', 'initialize_app'
]