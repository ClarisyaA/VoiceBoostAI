import pyaudio
import wave
import numpy as np
import streamlit as st
import threading
import time
import tempfile
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import random
import sounddevice as sd
import soundfile as sf
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa

class AudioRecorder:
    def __init__(self, sample_rate=22050, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.recording = False
        
    def start_recording(self):
        """Start recording audio"""
        self.frames = []
        self.recording = True
        
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            st.info("🎙️ Recording started... Speak now!")
            
            while self.recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            st.success("✅ Recording stopped!")
            
        except Exception as e:
            st.error(f"Error during recording: {e}")
            return None
    
    def stop_recording(self):
        """Stop recording audio"""
        self.recording = False
    
    def save_recording(self, filename=None):
        """Save recorded audio to file"""
        if not self.frames:
            return None
            
        if filename is None:
            # Create temporary file
            os.makedirs("recordings", exist_ok=True)
            filename = os.path.join("recordings", f"recording_{int(time.time())}.wav")
        
        try:
            # Save to WAV file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            return filename
            
        except Exception as e:
            st.error(f"Error saving recording: {e}")
            return None
    
    def get_audio_duration(self):
        """Get duration of recorded audio"""
        if not self.frames:
            return 0
        
        total_frames = len(self.frames) * self.chunk_size
        duration = total_frames / self.sample_rate
        return duration
    
    def cleanup(self):
        """Clean up PyAudio resources"""
        try:
            self.audio.terminate()
        except:
            pass

def visualize_audio(audio_file):
    """Create audio waveform visualization"""
    try:
        # Read audio file
        sample_rate, audio_data = wavfile.read(audio_file)
        
        # Create time axis
        duration = len(audio_data) / sample_rate
        time_axis = np.linspace(0, duration, len(audio_data))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Waveform plot
        ax1.plot(time_axis, audio_data, color='blue', alpha=0.7)
        ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram
        ax2.specgram(audio_data, Fs=sample_rate, cmap='viridis')
        ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Frequency (Hz)')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating audio visualization: {e}")
        return None

def audio_quality_check(audio_file):
    """Check audio quality and provide feedback"""
    try:
        sample_rate, audio_data = wavfile.read(audio_file)
        
        # Convert to float for analysis
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Calculate audio statistics
        duration = len(audio_data) / sample_rate
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        
        # Signal-to-noise ratio estimation
        noise_floor = np.percentile(np.abs(audio_data), 10)
        snr = 20 * np.log10(rms / (noise_floor + 1e-10))
        
        # Dynamic range
        dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
        
        quality_report = {
            'duration': duration,
            'sample_rate': sample_rate,
            'rms_level': rms,
            'peak_level': peak,
            'snr': snr,
            'dynamic_range': dynamic_range,
            'clipping': peak >= 0.99,
            'too_quiet': rms < 0.01,
            'good_quality': snr > 20 and not (peak >= 0.99) and rms > 0.01
        }
        
        return quality_report
        
    except Exception as e:
        st.error(f"Error checking audio quality: {e}")
        return None

def display_quality_report(quality_report):
    """Display audio quality report in Streamlit"""
    if quality_report is None:
        return
    
    st.subheader("🔍 Audio Quality Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Duration", f"{quality_report['duration']:.2f}s")
        st.metric("Sample Rate", f"{quality_report['sample_rate']} Hz")
    
    with col2:
        st.metric("RMS Level", f"{quality_report['rms_level']:.3f}")
        st.metric("Peak Level", f"{quality_report['peak_level']:.3f}")
    
    with col3:
        st.metric("SNR", f"{quality_report['snr']:.1f} dB")
        st.metric("Dynamic Range", f"{quality_report['dynamic_range']:.1f} dB")
    
    # Quality indicators
    st.subheader("Quality Indicators")
    
    if quality_report['good_quality']:
        st.success("✅ Good audio quality detected!")
    else:
        st.warning("⚠️ Audio quality issues detected:")
        
        if quality_report['clipping']:
            st.error("🔴 Audio clipping detected - reduce recording volume")
        
        if quality_report['too_quiet']:
            st.error("🔴 Audio too quiet - increase recording volume")
        
        if quality_report['snr'] < 20:
            st.warning("🟡 Low signal-to-noise ratio - record in quieter environment")

def convert_audio_format(input_file, output_format='wav', sample_rate=22050):
    """Convert audio to different format"""
    try:
        import librosa
        import soundfile as sf
        
        # Load audio
        y, sr = librosa.load(input_file, sr=sample_rate)
        
        # Create output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.{output_format}"
        
        # Save in new format
        sf.write(output_file, y, sample_rate)
        
        return output_file
        
    except Exception as e:
        st.error(f"Error converting audio format: {e}")
        return None

def enhance_audio(audio_file):
    """Apply basic audio enhancement"""
    try:
        import librosa
        import soundfile as sf
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Apply noise reduction (simple high-pass filter)
        y_filtered = librosa.effects.preemphasis(y)
        
        # Normalize audio
        y_normalized = librosa.util.normalize(y_filtered)
        
        # Save enhanced audio
        enhanced_file = audio_file.replace('.wav', '_enhanced.wav')
        sf.write(enhanced_file, y_normalized, sr)
        
        return enhanced_file
        
    except Exception as e:
        st.error(f"Error enhancing audio: {e}")
        return None

class RecordingSession:
    """Manage recording session with UI controls"""
    
    def __init__(self):
        self.recorder = None
        self.recording_thread = None
        
    def create_recording_interface(self):
        """Create Streamlit interface for recording"""
        st.subheader("🎙️ Voice Recording")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔴 Start Recording", type="primary"):
                self.start_recording_session()
        
        with col2:
            if st.button("⏹️ Stop Recording", type="secondary"):
                self.stop_recording_session()
        
        with col3:
            if st.button("🎵 Test Microphone", type="secondary"):
                self.test_microphone()
        
        # Recording status
        if 'recording_status' in st.session_state:
            if st.session_state.recording_status == 'recording':
                st.info("🎙️ Recording in progress...")
                
                # Real-time duration counter
                if 'recording_start_time' in st.session_state:
                    duration = time.time() - st.session_state.recording_start_time
                    st.metric("Recording Duration", f"{duration:.1f}s")
            
            elif st.session_state.recording_status == 'stopped':
                st.success("✅ Recording completed!")
                
                if 'last_recording' in st.session_state:
                    # Display audio player
                    st.audio(st.session_state.last_recording)
                    
                    # Quality analysis
                    quality_report = audio_quality_check(st.session_state.last_recording)
                    if quality_report:
                        display_quality_report(quality_report)
                    
                    # Audio visualization
                    if st.checkbox("Show Audio Visualization"):
                        fig = visualize_audio(st.session_state.last_recording)
                        if fig:
                            st.pyplot(fig)
    
    def start_recording_session(self):
        """Start recording session"""
        try:
            self.recorder = AudioRecorder()
            st.session_state.recording_status = 'recording'
            st.session_state.recording_start_time = time.time()
            
            # Start recording in separate thread
            self.recording_thread = threading.Thread(
                target=self.recorder.start_recording
            )
            self.recording_thread.start()
            
        except Exception as e:
            st.error(f"Error starting recording: {e}")
    
    def stop_recording_session(self):
        """Stop recording session"""
        if self.recorder and self.recorder.recording:
            self.recorder.stop_recording()
            
            if self.recording_thread:
                self.recording_thread.join()
            
            # Save recording
            filename = self.recorder.save_recording()
            if filename:
                st.session_state.last_recording = filename
                st.session_state.recording_status = 'stopped'
            
            # Cleanup
            self.recorder.cleanup()
    
    def test_microphone(self):
        """Test microphone functionality"""
        try:
            # Quick recording test
            test_recorder = AudioRecorder()
            
            with st.spinner("Testing microphone..."):
                # Record for 2 seconds
                test_recorder.recording = True
                stream = test_recorder.audio.open(
                    format=test_recorder.format,
                    channels=test_recorder.channels,
                    rate=test_recorder.sample_rate,
                    input=True,
                    frames_per_buffer=test_recorder.chunk_size
                )
                
                frames = []
                for _ in range(int(test_recorder.sample_rate / test_recorder.chunk_size * 2)):
                    data = stream.read(test_recorder.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                
                stream.stop_stream()
                stream.close()
                test_recorder.cleanup()
            
            # Analyze test recording
            if frames:
                # Convert to numpy array for analysis
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                
                if rms > 100:  # Arbitrary threshold
                    st.success("✅ Microphone is working properly!")
                    st.info(f"Signal level: {rms:.0f}")
                else:
                    st.warning("⚠️ Microphone signal is very low. Check your microphone settings.")
            else:
                st.error("❌ No audio signal detected. Check microphone connection.")
                
        except Exception as e:
            st.error(f"Microphone test failed: {e}")



def get_random_prompt(prompt_type="Word"):
    words = ["apple", "banana", "doctor", "river", "school"]
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "I am going to the market.",
        "Technology is evolving every day.",
        "Please speak clearly for this test."
    ]
    
    if prompt_type.lower() == "word":
        return random.choice(words)
    else:
        return random.choice(sentences)


# Helper functions for integration with main app
def get_audio_recording_component():
    """Get audio recording component for main app"""
    if 'recording_session' not in st.session_state:
        st.session_state.recording_session = RecordingSession()
    
    return st.session_state.recording_session

def record_audio(file_path, duration=5, fs=22050):
    """Rekam audio dari mic dan simpan ke file_path"""
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    sf.write(file_path, audio, fs)
    print("Recording saved to", file_path)

def extract_features(audio_path):
    """Ekstraksi fitur audio: formant, prosodik, dan spektral"""

    # === FORMANT FEATURES ===
    try:
        sound = parselmouth.Sound(audio_path)
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)

        f1_values, f2_values, f3_values = [], [], []

        for t in np.arange(0.1, sound.duration - 0.1, 0.05):
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            if not (np.isnan(f1) or np.isnan(f2) or np.isnan(f3)):
                f1_values.append(f1)
                f2_values.append(f2)
                f3_values.append(f3)

        f1_mean = np.mean(f1_values) if f1_values else 0
        f2_mean = np.mean(f2_values) if f2_values else 0
        f3_mean = np.mean(f3_values) if f3_values else 0
        vsa = (max(f1_values) - min(f1_values)) * (max(f2_values) - min(f2_values)) if f1_values and f2_values else 0
    except:
        f1_mean = f2_mean = f3_mean = vsa = 0

    # === PROSODIC FEATURES ===
    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        intensity = call(sound, "To Intensity", 75, 0.0, "yes")

        pitch_values = [call(pitch, "Get value at time", t, "Hertz", "Linear") 
                        for t in np.arange(0.1, sound.duration - 0.1, 0.01)]
        pitch_values = [p for p in pitch_values if not np.isnan(p)]

        intensity_values = [call(intensity, "Get value at time", t, "dB") 
                            for t in np.arange(0.1, sound.duration - 0.1, 0.01)]
        intensity_values = [i for i in intensity_values if not np.isnan(i)]

        pitch_range = max(pitch_values) - min(pitch_values) if pitch_values else 0
        intensity_range = max(intensity_values) - min(intensity_values) if intensity_values else 0
    except:
        pitch_range = intensity_range = 0

    # === SPECTRAL FEATURES ===
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        speaking_rate = len(onset_frames) / (len(y) / sr) * 60 if y is not None else 0
    except:
        speaking_rate = 0

    return {
        "f1_mean": f1_mean,
        "f2_mean": f2_mean,
        "f3_mean": f3_mean,
        "vsa": vsa,
        "pitch_range": pitch_range,
        "intensity_range": intensity_range,
        "speaking_rate": speaking_rate
    }