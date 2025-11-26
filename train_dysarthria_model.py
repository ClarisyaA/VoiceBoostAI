import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump  
from scipy.stats import kurtosis, skew
import parselmouth
from parselmouth.praat import call
import warnings
warnings.filterwarnings('ignore')   
import whisper
from textdistance import levenshtein
import re
from scipy.signal import wiener


class DysarthriaClassifier:
    def __init__(self, data_training_path, data_testing_path, model_save_path):
        self.data_training_path = data_training_path
        self.data_testing_path = data_testing_path
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        self.model = None
        self.whisper_model = whisper.load_model("base")

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
    
    def calculate_word_error_rate(self, audio_path, ground_truth_text):
        """Calculate Word Error Rate - FIXED VERSION"""
        try:
            # Transcribe dengan parameter yang lebih sensitif
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                temperature=0.0,
                no_speech_threshold=0.3,  # Lebih sensitif untuk menangkap speech
                logprob_threshold=-1.5,   # Lebih permisif
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False
            )
            
            recognized_text = result["text"].strip().lower()
            
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

    def extract_jitter_shimmer(self, audio_path):
        """Extract Jitter dan Shimmer - FIXED VERSION"""
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
        
    def extract_formant_features(self, audio_path):
        """Extract formant frequencies F1, F2, F3 using Parselmouth/Praat"""
        try:
            sound = parselmouth.Sound(audio_path)
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            f1_values = []
            f2_values = []
            f3_values = []
            
            # Extract formants at multiple time points
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
                    'vsa': self.calculate_vsa(f1_values, f2_values)  # Vowel Space Area
                }
        except:
            pass
            
        return {
            'f1_mean': 0, 'f1_std': 0, 'f2_mean': 0, 
            'f2_std': 0, 'f3_mean': 0, 'f3_std': 0, 'vsa': 0
        }
    
    def calculate_vsa(self, f1_values, f2_values):
        """Calculate Vowel Space Area - improved method"""
        if len(f1_values) < 3 or len(f2_values) < 3:
            return 5000
              # Default VSA for normal speech
        
        # Use percentile-based calculation to reduce outlier effects
        f1_range = np.percentile(f1_values, 90) - np.percentile(f1_values, 10)
        f2_range = np.percentile(f2_values, 90) - np.percentile(f2_values, 10)
        return f1_range * f2_range
    
    def extract_prosodic_features(self, audio_path):
        pitch_values = []
        intensity_values = []

        try:
            sound = parselmouth.Sound(audio_path)
            print(f"[DEBUG] Durasi audio: {sound.duration:.2f} detik")

            if sound.duration < 0.25:
                raise ValueError("Audio terlalu pendek untuk ekstraksi prosodi.")

            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            intensity = call(sound, "To Intensity", 75, 0.0, "yes")

            for t in np.arange(0.1, sound.duration - 0.1, 0.01):
                p = call(pitch, "Get value at time", t, "Hertz", "linear")
                if not np.isnan(p):
                    pitch_values.append(p)

            for t in np.arange(0.1, sound.duration - 0.1, 0.01):
                i = call(intensity, "Get value at time", t, "linear")
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
            print("❌ Error extracting prosodic features:", e)
            import traceback
            traceback.print_exc()

        return {
            'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0,
            'intensity_mean': 0, 'intensity_std': 0, 'intensity_range': 0
        }
    
    def extract_spectral_features(self, audio_path):
        """Extract spectral features using librosa with preprocessing"""
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
            print(f"Error extracting spectral features from {audio_path}: {e}")
            # Return default values
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
        """Extract all features from audio file - ENHANCED VERSION"""
        print(f"Processing: {audio_path}")
        
        # Original features
        formant_features = self.extract_formant_features(audio_path)
        prosodic_features = self.extract_prosodic_features(audio_path)
        spectral_features = self.extract_spectral_features(audio_path)
        
        # NEW FEATURES
        jitter_shimmer_features = self.extract_jitter_shimmer(audio_path)
        duration_features = self.extract_duration_features(audio_path)
        voice_quality_features = self.extract_voice_quality_features(audio_path)
        
        # WER features (cek jika ada file .txt)
        audio_dir = os.path.dirname(audio_path)
        session_dir = os.path.dirname(audio_dir)
        prompts_dir = os.path.join(session_dir, 'prompts')
        audio_filename = os.path.basename(audio_path).replace('.wav', '.txt')
        txt_path = os.path.join(prompts_dir, audio_filename)
        wer_features = {}
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                ground_truth_text = f.read().strip()
            wer_features = self.calculate_word_error_rate(audio_path, ground_truth_text)
        else:
            wer_features = {'word_error_rate': 0, 'word_count': 0}
        
        # Combine all features
        all_features = {
            **formant_features, 
            **prosodic_features, 
            **spectral_features,
            **jitter_shimmer_features,
            **duration_features,
            **voice_quality_features,
            **wer_features
        }
        
        return all_features
    
    def load_dataset(self):
        """Load and process dataset from both training and testing folders"""
        data = []
        labels = []
        
        # Process both training and testing data
        for data_path in [self.data_training_path, self.data_testing_path]:
            if not os.path.exists(data_path):
                print(f"Warning: Path {data_path} does not exist")
                continue
                
            print(f"\nProcessing data from: {data_path}")
            
            # Process dysarthria samples
            dysarthria_path = os.path.join(data_path, 'dysarthria')
            if os.path.exists(dysarthria_path):
                print("Processing dysarthria samples...")
                dysarthria_features = self.process_category(dysarthria_path)
                data.extend(dysarthria_features)
                labels.extend([1] * len(dysarthria_features))  # 1 for dysarthria
            
            # Process normal samples
            normal_path = os.path.join(data_path, 'normal')
            if os.path.exists(normal_path):
                print("Processing normal samples...")
                normal_features = self.process_category(normal_path)
                data.extend(normal_features)
                labels.extend([0] * len(normal_features))  # 0 for normal
        
        print(f"\nTotal samples processed: {len(data)}")
        print(f"Dysarthria samples: {sum(labels)}")
        print(f"Normal samples: {len(labels) - sum(labels)}")
        
        return np.array(data), np.array(labels)
    
    def process_category(self, category_path):
        """Process all audio files in a category (dysarthria or normal)"""
        features_list = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    try:
                        features = self.extract_all_features(audio_path)
                        if features:
                            features_list.append(list(features.values()))
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")
                        continue
                        
        return features_list
    
    def train_model(self, X, y):
        """Train the classification model"""
        print("\n=== Training Model ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try different models
        models = {
            'Random Forest': RandomForestClassifier( n_estimators=50,  # Kurangi dari 100
                max_depth=10,     # Batasi depth
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced')
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Dysarthria']))
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        print(f"\nBest model: {best_name} with accuracy: {best_score:.4f}")
        self.model = best_model

        print(confusion_matrix(y_test, y_pred))
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_name': best_name,
            'accuracy': best_score
        }
        
        model_file = os.path.join(self.model_save_path, 'dysarthria_classifier.joblib')  # Ganti ekstensi
        dump(model_data, model_file)
        
        print(f"\nModel saved to: {model_file}")
        return best_score
    
    def run_training(self):
        """Main training pipeline"""
        print("=== Dysarthria Classification Training ===")
        print("Loading and processing dataset...")
        
        # Load dataset
        X, y = self.load_dataset()
        
        if len(X) == 0:
            print("No data found! Please check your data paths.")
            return
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        
        # Train model
        accuracy = self.train_model(X, y)
        
        print(f"\n=== Training Completed ===")
        print(f"Final Model Accuracy: {accuracy:.4f}")
        print(f"Model saved in: {self.model_save_path}")
        
        return accuracy

def main():
    # Set paths
    data_training_path = "data_training"
    data_testing_path = "data_testing" 
    model_save_path = "models"
    
    # Create classifier instance
    classifier = DysarthriaClassifier(
        data_training_path=data_training_path,
        data_testing_path=data_testing_path,
        model_save_path=model_save_path
    )
    
    # Run training
    classifier.run_training()

if __name__ == "__main__":
    main()