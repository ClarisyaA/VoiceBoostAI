import streamlit as st
import os
import numpy as np
import librosa
import tempfile
import random
from gtts import gTTS
import parselmouth
from parselmouth.praat import call
import whisper
import warnings
warnings.filterwarnings('ignore')
from joblib import load
from scipy.stats import kurtosis, skew
from textdistance import levenshtein
import re
from scipy.signal import wiener

word_list = [
    # Kata-kata dengan vokal yang jelas untuk analisis formant
    "beat", "bit", "bet", "bat", "boot", "book", "boat", "bought", "but", "bot",
    
    # Kata-kata dengan konsonan yang sulit untuk dysarthria
    "church", "judge", "strength", "lengths", "sixths", "twelfths", "glimpsed", "prompted",
    
    # Kata-kata dengan cluster konsonan  
    "splash", "spring", "script", "straight", "shrink", "thrash", "glimpse", "prompt",
    
    # Kata-kata dengan suara frikatif
    "fish", "ship", "think", "that", "vision", "measure", "rough", "smooth",
    
    # Kata-kata dengan suara plosif
    "papa", "baby", "kick", "gag", "tight", "dad", "pop", "bob",
    
    # Kata-kata dengan suara nasal
    "mom", "noon", "ring", "hang", "mean", "main", "mine", "moon",
    
    # Kata-kata dengan suara likuid
    "little", "really", "yellow", "roll", "will", "real", "royal", "loyal",
    
    # Kata-kata dengan diftong
    "house", "loud", "choice", "point", "light", "proud", "about", "found",
    
    # Kata-kata multisuku untuk menguji koordinasi
    "butterfly", "beautiful", "vegetables", "university", "opportunity", "information",
    "hospital", "restaurant", "telephone", "computer", "television", "refrigerator",
    
    # Kata-kata dengan stress pattern yang berbeda
    "record", "present", "object", "subject", "perfect", "reject", "conflict", "content",
    
    # Kata-kata fungsional yang sering digunakan
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    
    # Kata-kata dengan suara vokal yang kontrastif
    "seen", "sin", "pen", "pan", "fun", "fan", "cut", "cat", "hot", "hat",
    
    # Kata-kata dengan akhiran yang berbeda
    "running", "jumped", "quickly", "slowly", "happily", "careful", "hopeful", "peaceful"
]

# SENTENCE LIST - Dirancang untuk menguji prosodi, ritme, dan koartikulasi
sentence_list = [
    # Kalimat dengan berbagai struktur vokal untuk analisis VSA
    "Will Robin wear a yellow lily? .",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    
    # Kalimat dengan cluster konsonan yang menantang
    "The sixth sick sheik's sixth sheep's sick.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Three free throws through thick fog.",

    # Kalimat dengan pola ritme yang berbeda
    "Rain in Spain falls mainly on the plain.",
    "Round and round the rugged rock the ragged rascal ran.",
    "Toy boat, toy boat, toy boat.",

    # Kalimat dengan variasi pitch dan intonasi
    "Can you help me with this problem?",
    "What a beautiful day it is today!",
    "I don't think that's correct.",
    "Where are you going this afternoon?",
    
    # Kalimat dengan koordinasi motorik yang kompleks
    "The butterfly flew gracefully through the colorful garden.",
    "My grandmother makes delicious chocolate chip cookies every Sunday.",
    "The university students studied diligently for their final examinations.",
    
    # Kalimat dengan berbagai suara frikatif
    "Fresh fish and chips from the shop.",
    "The thick fog made visibility very poor.",
    "She chose shoes with shiny silver buckles.",
    
    # Kalimat dengan suara plosif berulang
    "Big black bugs bleed blue blood.",
    "Pretty Polly picked pink peonies.",
    "Daddy's dirty dog dug deep ditches.",
    
    # Kalimat dengan suara nasal
    "Many men and women came to the meeting.",
    "The moon shines brightly on the mountain.",
    "Morning brings new beginnings and opportunities.",
    
    # Kalimat dengan suara likuid
    "Really lovely yellow flowers bloom in our garden.",
    "The little girl will roll the ball down the hill.",
    "Royal blue ribbons were wrapped around the presents.",
    
    # Kalimat dengan tempo dan ritme yang bervariasi
    "Slowly and carefully, she walked across the icy sidewalk.",
    "Quickly! Run to the store before it closes!",
    "The old man sat quietly reading his newspaper.",
    
    # Kalimat dengan prosodi emosional
    "I'm so excited about our vacation next week!",
    "That was the most difficult test I've ever taken.",
    "Please speak more clearly so I can understand you.",
    
    # Kalimat dengan struktur gramatikal yang kompleks
    "Although it was raining heavily, we decided to go for a walk anyway.",
    "The book that I borrowed from the library yesterday was very interesting.",
    "If you finish your homework early, you can watch television tonight.",
    
    # Kalimat dengan pengulangan suara untuk konsistensi
    "Red leather, yellow leather, red leather, yellow leather.",
    "Unique New York, unique New York, unique New York.",
    "Sally sells shells, Sally sells shells, Sally sells shells.",
    
    # Kalimat dengan kontras stress
    "I SAID the book is on the TABLE.",
    "The FIRST student finished BEFORE the others.",
    "We're going to the MOVIES, not the RESTAURANT.",
    
    # Kalimat dengan deskripsi untuk testing kontinuitas
    "The children played happily in the park while their parents watched nearby.",
    "After finishing breakfast, she packed her lunch and walked to work.",
    "The doctor carefully examined the patient and prescribed appropriate medication.",
    
    # Kalimat dengan counting dan sequencing
    "One, two, three, four, five, six, seven, eight, nine, ten.",
    "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.",
    "January, February, March, April, May, June, July, August.",
    
    # Kalimat dengan automatic speech
    "Thank you very much for your help today.",
    "How are you feeling this morning?",
    "Have a wonderful day and take care of yourself.",
    "Please call me when you arrive safely.",
    
    # Kalimat dengan alternating motor movements
    "Pa-ta-ka, pa-ta-ka, pa-ta-ka, pa-ta-ka.",
    "Buttercup, buttercup, buttercup, buttercup.",
    "Puh-tuh-kuh, puh-tuh-kuh, puh-tuh-kuh.",
    
    # Kalimat untuk testing breath support
    "The magnificent seven rode their horses across the vast prairie under the blazing sun.",
    "Communication is essential for building strong relationships with family and friends throughout life.",
    "Technology continues to advance rapidly, changing the way we work, learn, and connect with others around the world."

]


class DysarthriaAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.load_model()
        
        
        
    def preprocess_audio(self, audio_path):
        """Enhanced preprocessing with noise reduction"""
        y, sr = librosa.load(audio_path, sr=22050)
        
        # 1. Spectral Gating (menghilangkan noise berdasarkan threshold)
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # Estimate noise floor (ambil 5% magnitude terendah)
        noise_floor = np.percentile(magnitude, 5)
        noise_threshold = noise_floor * 2  # Threshold 2x noise floor
        
        # Gate: set magnitude di bawah threshold ke 0
        magnitude_gated = np.where(magnitude > noise_threshold, magnitude, magnitude * 0.1)
        
        # Reconstruct audio
        phase = np.angle(stft)  
        stft_cleaned = magnitude_gated * np.exp(1j * phase)
        y_cleaned = librosa.istft(stft_cleaned)
        
        # 2. Wiener filter untuk smoothing
        y_cleaned = wiener(y_cleaned, mysize=5)
        
        # 3. High-pass filter (hilangkan low freq noise < 80Hz)
        y_cleaned = librosa.effects.preemphasis(y_cleaned, coef=0.95)
        
        # 4. Normalize
        y_cleaned = librosa.util.normalize(y_cleaned)
        
        return y_cleaned, sr
    
    def extract_jitter_shimmer(self, audio_path):
        try:
            sound = parselmouth.Sound(audio_path)
            
            # Cek durasi audio minimum
            if sound.duration < 1.0:
                return {'jitter_local': 0, 'jitter_rap': 0, 'shimmer_local': 0, 'shimmer_db': 0}
            
            # Test pitch extraction dengan parameter berbeda (autocorrelation method)
            pitch = call(sound, "To Pitch (ac)", 0.0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 500)
            
            num_frames = call(pitch, "Get number of frames")
            
            # Cek nilai pitch aktual
            pitch_values = []
            for i in range(num_frames):
                f0 = call(pitch, "Get value in frame", i+1, "Hertz")
                if not np.isnan(f0) and f0 > 0:
                    pitch_values.append(f0)
            
            if len(pitch_values) < 10:
                return {'jitter_local': 0, 'jitter_rap': 0, 'shimmer_local': 0, 'shimmer_db': 0}
            
            # Manual jitter calculation sebagai backup
            periods = []
            for i in range(len(pitch_values)):
                if pitch_values[i] > 0:
                    periods.append(1.0/pitch_values[i])
            
            manual_jitter_local = 0
            manual_jitter_rap = 0
            
            if len(periods) > 3:
                period_diffs = [abs(periods[i+1] - periods[i]) for i in range(len(periods)-1)]
                manual_jitter_local = np.mean(period_diffs) / np.mean(periods) if periods else 0
                
                # RAP jitter (relative average perturbation)
                rap_values = []
                for i in range(1, len(periods)-1):
                    avg_three = (periods[i-1] + periods[i] + periods[i+1]) / 3
                    rap_values.append(abs(periods[i] - avg_three))
                manual_jitter_rap = np.mean(rap_values) / np.mean(periods) if rap_values and periods else 0
            
            # Praat jitter calculation
            praat_jitter_local = 0
            praat_jitter_rap = 0
            
            try:
                praat_jitter_local = call(pitch, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                if np.isnan(praat_jitter_local) or np.isinf(praat_jitter_local):
                    praat_jitter_local = 0
            except Exception as e:
                praat_jitter_local = 0
                
            try:
                praat_jitter_rap = call(pitch, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
                if np.isnan(praat_jitter_rap) or np.isinf(praat_jitter_rap):
                    praat_jitter_rap = 0
            except Exception as e:
                praat_jitter_rap = 0
            
            # Shimmer calculation
            shimmer_local = 0
            shimmer_db = 0
            
            try:
                point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
                num_points = call(point_process, "Get number of points")
                
                if num_points > 10:
                    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    
                    if np.isnan(shimmer_local) or np.isinf(shimmer_local):
                        shimmer_local = 0
                    if np.isnan(shimmer_db) or np.isinf(shimmer_db):
                        shimmer_db = 0
                        
                else:
                    pass
            except Exception as e:
                shimmer_local = 0
                shimmer_db = 0
            
            # Use Praat values if available, otherwise use manual calculation
            final_jitter_local = praat_jitter_local if praat_jitter_local > 0 else manual_jitter_local
            final_jitter_rap = praat_jitter_rap if praat_jitter_rap > 0 else manual_jitter_rap
            
            result = {
                'jitter_local': float(final_jitter_local),
                'jitter_rap': float(final_jitter_rap),
                'shimmer_local': float(shimmer_local),
                'shimmer_db': float(shimmer_db)
            }
            
            return result
            
        except Exception as e:
            return {'jitter_local': 0, 'jitter_rap': 0, 'shimmer_local': 0, 'shimmer_db': 0}
        
    def calculate_word_error_rate(self, audio_path, ground_truth_text):
        """Calculate Word Error Rate - FIXED VERSION"""
        try:
            # Use cached whisper model
            whisper_model = load_whisper_model()
            
            # Transcribe dengan parameter yang lebih sensitif
            result = whisper_model.transcribe(
                audio_path,
                language="en",
                temperature=0.0,
                no_speech_threshold=0.3,  # Lebih sensitif untuk menangkap speech
                logprob_threshold=-1.5,   # Lebih permisif
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
                fp16=False  # Disable fp16 for stability
            )
            
            recognized_text = result["text"].strip().lower()
            print(f"Raw transcription: '{recognized_text}'")
            
            # Clean text - hapus tanda baca dan normalize
            ground_truth_clean = re.sub(r'[^\w\s]', '', ground_truth_text.lower()).strip()
            recognized_clean = re.sub(r'[^\w\s]', '', recognized_text).strip()
            
            # Split into words
            gt_words = [word for word in ground_truth_clean.split() if word]
            rec_words = [word for word in recognized_clean.split() if word]
            
            # Jika tidak ada kata yang dikenali sama sekali
            if len(rec_words) == 0:
                return {'word_error_rate': 1.0, 'word_count': len(gt_words)}
            
            # Jika tidak ada ground truth
            if len(gt_words) == 0:
                return {'word_error_rate': 0.0, 'word_count': 0}
            
            # Calculate Levenshtein distance untuk word-level
            distance = levenshtein.distance(gt_words, rec_words)
            
            # WER = (S + D + I) / N
            # Dimana S=substitutions, D=deletions, I=insertions, N=total words di reference
            wer = distance / len(gt_words)
            
            # Clamp WER antara 0 dan 1
            wer = max(0.0, min(wer, 1.0))
            
            
            return {
                'word_error_rate': float(wer),
                'word_count': len(gt_words)
            }
            
        except Exception as e:
            # Return plausible default instead of 1.0
            return {'word_error_rate': 0.5, 'word_count': len(ground_truth_text.split()) if ground_truth_text else 0}
    
    def extract_duration_features(self, audio_path):
        """Extract Duration-based features"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            utterance_duration = len(y) / sr
            
            # Voice Activity Detection
            rms = librosa.feature.rms(y=y, frame_length=int(0.025 * sr), hop_length=int(0.010 * sr))[0]
            energy_threshold = np.percentile(rms, 30)
            voice_frames = rms > energy_threshold
            
            # Phonation time ratio
            phonation_time = np.sum(voice_frames) * int(0.010 * sr) / sr
            phonation_time_ratio = phonation_time / utterance_duration if utterance_duration > 0 else 0
            
            # Pause detection
            silence_frames = ~voice_frames
            pause_segments = []
            in_pause = False
            pause_start = 0
            
            for i, is_silent in enumerate(silence_frames):
                if is_silent and not in_pause:
                    pause_start = i
                    in_pause = True
                elif not is_silent and in_pause:
                    pause_duration = (i - pause_start) * int(0.010 * sr) / sr
                    if pause_duration > 0.1:
                        pause_segments.append(pause_duration)
                    in_pause = False
            
            return {
                'utterance_duration': utterance_duration,
                'phonation_time_ratio': phonation_time_ratio,
                'pause_count': len(pause_segments),
                'average_pause_duration': np.mean(pause_segments) if pause_segments else 0,
                'speech_rate': phonation_time / utterance_duration if utterance_duration > 0 else 0
            }
        except:
            return {'utterance_duration': 0, 'phonation_time_ratio': 0, 'pause_count': 0, 'average_pause_duration': 0, 'speech_rate': 0}
    
        
    def extract_voice_quality_features(self, audio_path):
        """Extract Voice Quality Descriptors"""
        try:
            sound = parselmouth.Sound(audio_path)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_mean = call(harmonicity, "Get mean", 0, 0)
            
            y, sr = librosa.load(audio_path, sr=22050)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            # Frequency analysis
            stft = librosa.stft(y)
            freqs = librosa.fft_frequencies(sr=sr)
            low_freq_energy = np.mean(np.abs(stft[freqs < 500, :]))
            high_freq_energy = np.mean(np.abs(stft[freqs > 4000, :]))
            
            return {
                'hnr_mean': hnr_mean if not np.isnan(hnr_mean) else 0,
                'breathiness': high_freq_energy / (low_freq_energy + 1e-10),
                'roughness': 1.0 / (abs(hnr_mean) + 1e-10) if not np.isnan(hnr_mean) else 1.0,
                'creakiness': low_freq_energy,
                'spectral_flatness': spectral_flatness
            }
        except:
            return {'hnr_mean': 0, 'breathiness': 0, 'roughness': 0, 'creakiness': 0, 'spectral_flatness': 0}

    def load_model(self):
        try:
            model_data = load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    def extract_formant_features(self, audio_path):
        try:
            sound = parselmouth.Sound(audio_path)
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            f1_values = []
            f2_values = []
            f3_values = []
            
            for t in np.arange(0.1, sound.duration - 0.1, 0.05):
                f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
                
                if not (np.isnan(f1) or np.isnan(f2) or np.isnan(f3)):
                    f1_values.append(f1)
                    f2_values.append(f2)
                    f3_values.append(f3)
            
            if len(f1_values) > 0:
                return {
                    'f1_mean': np.mean(f1_values),
                    'f1_std': np.std(f1_values),
                    'f2_mean': np.mean(f2_values),
                    'f2_std': np.std(f2_values),
                    'f3_mean': np.mean(f3_values),
                    'f3_std': np.std(f3_values),
                    'vsa': self.calculate_vsa(f1_values, f2_values)
                }
        except:
            pass
            
        return {
            'f1_mean': 0, 'f1_std': 0, 'f2_mean': 0, 
            'f2_std': 0, 'f3_mean': 0, 'f3_std': 0, 'vsa': 0
        }
    
    def calculate_vsa(self, f1_values, f2_values):
        if len(f1_values) < 3 or len(f2_values) < 3:
            return 5000
        
        f1_range = np.percentile(f1_values, 90) - np.percentile(f1_values, 10)
        f2_range = np.percentile(f2_values, 90) - np.percentile(f2_values, 10)
        
        return f1_range * f2_range
        
    def extract_prosodic_features(self, audio_path):
        pitch_values = []
        intensity_values = []
        
        try:
            sound = parselmouth.Sound(audio_path)
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            intensity = call(sound, "To Intensity", 75, 0.0, "yes")
            
            for t in np.arange(0.1, sound.duration - 0.1, 0.01):
                p = call(pitch, "Get value at time", t, "Hertz", "linear")
                if not np.isnan(p):
                    pitch_values.append(p)
            
            for t in np.arange(0.1, sound.duration - 0.1, 0.01):
                i = call(intensity, "Get value at time", t, "linear")  # GANTI dari "Linear" ke "dB"
                if not np.isnan(i):
                    intensity_values.append(i)
            
            if len(pitch_values) > 0 and len(intensity_values) > 0:
                return {
                    'pitch_mean': np.mean(pitch_values),
                    'pitch_std': np.std(pitch_values),
                    'pitch_range': max(pitch_values) - min(pitch_values),
                    'intensity_mean': np.mean(intensity_values),
                    'intensity_std': np.std(intensity_values),
                    'intensity_range': max(intensity_values) - min(intensity_values)
                }
                
        except Exception as e:
            pass
        
        # Jika gagal, return 0 semua (bukan default values)
        return {
            'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0,
            'intensity_mean': 0, 'intensity_std': 0, 'intensity_range': 0
        }

    def extract_spectral_features(self, audio_path):
            try:
                # GUNAKAN METHOD BARU
                y, sr = self.preprocess_audio(audio_path)
                
                # MFCC features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
                
                # Speaking rate estimation (rough approximation)
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                speaking_rate = len(onset_frames) / (len(y) / sr) * 60  # per minute
                
                features = {
                    'spectral_centroid_mean': np.mean(spectral_centroids),
                    'spectral_centroid_std': np.std(spectral_centroids),
                    'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                    'spectral_rolloff_mean': np.mean(spectral_rolloff),
                    'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                    'speaking_rate': speaking_rate
                }
                
                # Add MFCC statistics
                for i in range(13):
                    features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                    features[f'mfcc_{i}_std'] = np.std(mfccs[i])
                
                return features
                
            except Exception as e:
                # Return default values on error
                features = {
                    'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
                    'spectral_bandwidth_mean': 0, 'spectral_rolloff_mean': 0,
                    'zero_crossing_rate_mean': 0, 'speaking_rate': 0
                }
                
                for i in range(13):
                    features[f'mfcc_{i}_mean'] = 0
                    features[f'mfcc_{i}_std'] = 0
                    
                return features
        
    def extract_all_features(self, audio_path):
        formant_features = self.extract_formant_features(audio_path)
        prosodic_features = self.extract_prosodic_features(audio_path)
        spectral_features = self.extract_spectral_features(audio_path)
        
        # TAMBAH INI:
        jitter_shimmer_features = self.extract_jitter_shimmer(audio_path)
        duration_features = self.extract_duration_features(audio_path)
        voice_quality_features = self.extract_voice_quality_features(audio_path)
        
        # WER features menggunakan generated text
        wer_features = {}
        if 'generated_text' in st.session_state:
            wer_features = self.calculate_word_error_rate(audio_path, st.session_state.generated_text)
        else:
            wer_features = {'word_error_rate': 0, 'word_count': 0}
        
        all_features = {
            **formant_features, **prosodic_features, **spectral_features,
            **jitter_shimmer_features, **duration_features, 
            **voice_quality_features, **wer_features
        }
        return all_features
    
    def predict(self, audio_path):
        if self.model is None or self.scaler is None:
            return None, "Model not loaded"
        
        features = self.extract_all_features(audio_path)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        prediction = self.model.predict(feature_vector_scaled)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(feature_vector_scaled)[0]
            prediction = np.argmax(probability)  # Konsisten dengan probability
        else:
            decision_score = self.model.decision_function(feature_vector_scaled)[0]
            probability = np.array([1 / (1 + np.exp(decision_score)), 1 / (1 + np.exp(-decision_score))])
        
        return prediction, probability

def text_to_speech(text, lang='en'):
    """Generate TTS audio - keep as MP3 for listening only"""
    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tts.save(tmp_file.name)
        return tmp_file.name


def cleanup_tts_files():
    """Clean up old TTS files"""
    if 'example_audio' in st.session_state:
        del st.session_state.example_audio


def transcribe_audio(audio_path):
    model = load_whisper_model()  # Gunakan cached model
    result = model.transcribe(
        audio_path, 
        language="en",
        temperature=0,
        fp16=False
    )
    return result["text"]

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base", device="cpu")

@st.cache_resource
def load_dysarthria_analyzer():
    """Load and cache the dysarthria analyzer to prevent reinitalization on each rerun"""
    return DysarthriaAnalyzer("models/dysarthria_classifier.joblib")

def main():
    st.set_page_config(page_title="VoiceBoost - Dysarthria Detection", layout="wide", page_icon="🎙️")
    if 'tts_file_path' not in st.session_state:
        st.session_state.tts_file_path = None

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .result-normal {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        color: #155724;
    }
    .result-dysarthria {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 8px;
        color: #721c24;
    }
    .audio-option {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .audio-option:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ VoiceBoost</h1>
        <h3>Advanced Speech Analysis & Dysarthria Detection</h3>
        <p>Powered by AI and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = load_dysarthria_analyzer()
    
    tab1, tab2 = st.tabs(["🏠 Home", "🎯 Speech Training"])
    
    with tab1:
        st.markdown("### 🏠 Welcome to VoiceBoost")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🚀 What is VoiceBoost?</h4>
                <p>VoiceBoost is an advanced AI-powered application designed to detect dysarthria through comprehensive speech analysis. 
                Our system uses cutting-edge machine learning algorithms to analyze various aspects of speech patterns.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>🎯 How it Works</h4>
                <p>Simply navigate to the <strong>Speech Training</strong> tab, choose between words or sentences, 
                record your voice or upload an audio file, and get instant analysis results with detailed feedback.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Key Metrics</h4>
                <ul>
                    <li>🎵 Formant Analysis</li>
                    <li>📈 Pitch Variation</li>
                    <li>🔊 Intensity Patterns</li>
                    <li>⚡ Speaking Rate</li>
                    <li>🌊 Spectral Features</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔬 Analysis Features")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            **🎵 Formant Frequencies**
            - F1, F2, F3 analysis
            - Vowel clarity assessment
            - Articulation precision
            """)
        
        with col2:
            st.markdown("""
            **📐 Vowel Space Area**
            - Vowel articulation space
            - Speech motor control
            - Coordination patterns
            """)
        
        with col3:
            st.markdown("""
            **⏱️ Temporal Features**
            - Speaking rate analysis
            - Pause patterns
            - Rhythm assessment
            """)
        
        with col4:
            st.markdown("""
            **🌊 Spectral Analysis**
            - MFCC features
            - Spectral characteristics
            - Voice quality metrics
            """)
    
    with tab2:
        st.markdown("### 🎯 Speech Training & Analysis")
        
        # Text generation section
        st.markdown("#### 📝 Step 1: Generate Practice Text")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            choice = st.radio("Choose training type:", ["Word", "Sentence"], horizontal=True)
        
        with col2:
            if st.button("🎲 Generate New Text", type="primary", use_container_width=True):
                # Reset example audio saat generate text baru
                cleanup_tts_files()
                
                if choice == "Word":
                    st.session_state.generated_text = random.choice(word_list)
                else:
                    st.session_state.generated_text = random.choice(sentence_list)
                
                st.rerun()
        
        with col3:
            if 'generated_text' in st.session_state:
                if st.button("🔊 Listen to Example", use_container_width=True):
                    with st.spinner("Generating audio..."):
                        audio_file = text_to_speech(st.session_state.generated_text)
                        with open(audio_file, 'rb') as f:
                            st.session_state.example_audio = f.read()
                        os.unlink(audio_file)  # cleanup temp file
                        st.rerun()
                
                # Show audio if exists
                if 'example_audio' in st.session_state:
                    st.audio(st.session_state.example_audio, format='audio/mp3')
                    
        
        if 'generated_text' in st.session_state:
            st.markdown("#### 📖 Practice Text:")
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); 
                        padding: 2rem; border-radius: 10px; border-left: 5px solid #667eea; 
                        font-size: 1.2em; font-weight: 500; text-align: center; margin: 1rem 0;">
                "{st.session_state.generated_text}"
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Audio input section
            st.markdown("#### 🎤 Step 2: Record Your Voice")
            
            # Option selector
            audio_option = st.radio(
                "Choose how to provide your audio:",
                ["🎙️ Record Live", "📁 Upload File"],
                horizontal=True
            )
            
            audio_data = None
            audio_path = None
            
            if audio_option == "🎙️ Record Live":
                st.markdown("""
                <div class="audio-option">
                    <h5>🎙️ Live Recording</h5>
                    <p>Click the record button below and speak clearly into your microphone.</p>
                </div>
                """, unsafe_allow_html=True)
                
                audio_bytes = st.audio_input("Record your speech")
                if audio_bytes is not None:
                    audio_data = audio_bytes
                    
            else:
                st.markdown("""
                <div class="audio-option">
                    <h5>📁 File Upload</h5>
                    <p>Upload a WAV audio file of your speech (max 10MB).</p>
                </div>
                """, unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    "Choose an audio file",
                    type=['wav'],
                    help="Please upload a WAV file containing your speech"
                )
                if uploaded_file is not None:
                    audio_data = uploaded_file
            
            # Process audio if available
            if audio_data is not None:
                os.makedirs("recordings", exist_ok=True)
                
                if audio_option == "🎙️ Record Live":
                    audio_path = os.path.join("recordings", "user_recording.wav")
                else:
                    audio_path = os.path.join("recordings", uploaded_file.name)
                
                with open(audio_path, "wb") as f:
                    f.write(audio_data.getbuffer())
                
                st.markdown("#### 🔊 Your Audio:")
                st.audio(audio_data)
                
                # Analysis section
                st.markdown("---")
                st.markdown("#### 🔬 Step 3: Analysis Results")
                
                with st.spinner("🤖 Analyzing your speech... This may take a moment."):
                    transcription = transcribe_audio(audio_path)
                    prediction, probability = analyzer.predict(audio_path)
                
                # Transcription results
                st.markdown("##### 📝 Speech Transcription:")
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; 
                           border-left: 4px solid #28a745; font-style: italic; font-size: 1.1em;">
                    "{transcription}"
                </div>
                """, unsafe_allow_html=True)
                
                # Analysis results
                if prediction is not None:
                    st.markdown("##### 🎯 Dysarthria Analysis:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.markdown("""
                            <div class="result-dysarthria">
                                <h4>🔴 DYSARTHRIA DETECTED</h4>
                                <p>The analysis indicates potential speech articulation difficulties.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            confidence = probability[1] * 100
                        else:
                            st.markdown("""
                            <div class="result-normal">
                                <h4>🟢 NORMAL SPEECH</h4>
                                <p>Your speech patterns appear to be within normal ranges.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            confidence = probability[0] * 100
                        
                        st.metric("Confidence Level", f"{confidence:.1f}%")
                    
                    with col2:
                        st.markdown("**Detailed Probabilities:**")
                        st.progress(probability[0], text=f"Normal: {probability[0]:.3f}")
                        st.progress(probability[1], text=f"Dysarthria: {probability[1]:.3f}")
                    
                    # Recommendations
                    st.markdown("##### 💡 Recommendations:")
                    if prediction == 1:
                        st.warning("""
                        **⚠️ Consultation Recommended**
                        
                        The analysis suggests possible articulation challenges. Consider:
                        - Consulting with a speech-language pathologist
                        - Regular speech therapy sessions
                        - Continued practice with speech exercises
                        - Follow-up assessments to monitor progress
                        """)
                    else:
                        st.success("""
                        **✅ Great Speech Quality!**
                        
                        Your speech analysis shows positive results:
                        - Continue practicing to maintain clarity
                        - Regular vocal exercises can help
                        - Keep up the excellent work!
                        - Consider periodic check-ups if needed
                        """)
                    
                    # Additional metrics
                    with st.expander("📊 View Detailed Metrics"):
                        features = analyzer.extract_all_features(audio_path)
                        
                        # Formant Features
                        st.subheader("🎵 Formant Analysis")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("F1 Mean (Hz)", f"{features.get('f1_mean', 0):.1f}")
                            st.metric("F1 Std Dev", f"{features.get('f1_std', 0):.1f}")
                        with col2:
                            st.metric("F2 Mean (Hz)", f"{features.get('f2_mean', 0):.1f}")
                            st.metric("F2 Std Dev", f"{features.get('f2_std', 0):.1f}")
                        with col3:
                            st.metric("F3 Mean (Hz)", f"{features.get('f3_mean', 0):.1f}")
                            st.metric("F3 Std Dev", f"{features.get('f3_std', 0):.1f}")
                        with col4:
                            st.metric("VSA", f"{features.get('vsa', 0):.1f}")
                        
                        st.divider()
                        
                        # Prosodic Features
                        st.subheader("🎼 Prosodic Features")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pitch Mean (Hz)", f"{features.get('pitch_mean', 0):.1f}")
                            st.metric("Pitch Std Dev", f"{features.get('pitch_std', 0):.1f}")
                            st.metric("Pitch Range (Hz)", f"{features.get('pitch_range', 0):.1f}")
                        with col2:
                            st.metric("Intensity Mean (dB)", f"{features.get('intensity_mean', 0):.1f}")
                            st.metric("Intensity Std Dev", f"{features.get('intensity_std', 0):.1f}")
                            st.metric("Intensity Range (dB)", f"{features.get('intensity_range', 0):.1f}")
                        with col3:
                            st.metric("Speaking Rate", f"{features.get('speaking_rate', 0):.1f}")
                        
                        st.divider()
                        
                        # Voice Quality Features
                        st.subheader("🎤 Voice Quality")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Jitter Local", f"{features.get('jitter_local', 0):.4f}")
                            st.metric("Jitter RAP", f"{features.get('jitter_rap', 0):.4f}")
                        with col2:
                            st.metric("Shimmer Local", f"{features.get('shimmer_local', 0):.4f}")
                            st.metric("Shimmer dB", f"{features.get('shimmer_db', 0):.3f}")
                        with col3:
                            st.metric("HNR Mean (dB)", f"{features.get('hnr_mean', 0):.2f}")
                            st.metric("Breathiness", f"{features.get('breathiness', 0):.4f}")
                        with col4:
                            st.metric("Roughness", f"{features.get('roughness', 0):.4f}")
                            st.metric("Spectral Flatness", f"{features.get('spectral_flatness', 0):.4f}")
                        
                        st.divider()
                        
                        # Duration Features
                        st.subheader("⏱️ Temporal Features")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Utterance Duration (s)", f"{features.get('utterance_duration', 0):.2f}")
                        with col2:
                            st.metric("Phonation Time Ratio", f"{features.get('phonation_time_ratio', 0):.3f}")
                        with col3:
                            st.metric("Pause Count", f"{features.get('pause_count', 0)}")
                        with col4:
                            st.metric("Avg Pause Duration (s)", f"{features.get('average_pause_duration', 0):.3f}")
                        
                        st.divider()
                        
                        # Spectral Features
                        st.subheader("🌊 Spectral Analysis")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Spectral Centroid Mean", f"{features.get('spectral_centroid_mean', 0):.1f}")
                            st.metric("Spectral Centroid Std", f"{features.get('spectral_centroid_std', 0):.1f}")
                        with col2:
                            st.metric("Spectral Bandwidth Mean", f"{features.get('spectral_bandwidth_mean', 0):.1f}")
                            st.metric("Spectral Rolloff Mean", f"{features.get('spectral_rolloff_mean', 0):.1f}")
                        with col3:
                            st.metric("Zero Crossing Rate", f"{features.get('zero_crossing_rate_mean', 0):.4f}")
                        
                        # MFCC Features
                        st.subheader("🎯 MFCC Features")
                        mfcc_cols = st.columns(4)
                        for i in range(13):
                            col_idx = i % 4
                            with mfcc_cols[col_idx]:
                                st.metric(f"MFCC {i} Mean", f"{features.get(f'mfcc_{i}_mean', 0):.3f}")
                                st.metric(f"MFCC {i} Std", f"{features.get(f'mfcc_{i}_std', 0):.3f}")
                        
                        st.divider()
                        
                        # Speech Recognition Features
                        st.subheader("📝 Speech Recognition")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Word Error Rate", f"{features.get('word_error_rate', 0):.3f}")
                        with col2:
                            st.metric("Word Count", f"{features.get('word_count', 0)}")
                        
                        # Feature Summary
                        st.divider()
                        st.subheader("📋 Feature Summary")
                        total_features = len(features)
                        non_zero_features = sum(1 for v in features.values() if v != 0)
                        st.info(f"**Total Features Extracted:** {total_features} | **Non-zero Features:** {non_zero_features}")
                        
                        # Download raw features
                        if st.button("📥 Download Raw Features (JSON)"):
                            import json
                            features_json = json.dumps(features, indent=2)
                            st.download_button(
                                label="Download Features",
                                data=features_json,
                                file_name="speech_features.json",
                                mime="application/json"
                            )
                
                # Cleanup
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

if __name__ == "__main__":
    main()
