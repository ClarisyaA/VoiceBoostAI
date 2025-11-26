# VoiceBoost - Dysarthria Classification System

Sistem AI untuk mengklasifikasi gangguan bicara dysarthria menggunakan machine learning dan analisis suara.

## 🚀 Fitur Utama

- **Speech Training**: Latihan bicara dengan kata dan kalimat
- **AI Classification**: Klasifikasi otomatis dysarthria vs normal
- **Real-time Recording**: Rekam suara langsung dari browser
- **Feature Analysis**: Analisis mendalam 13+ fitur suara
- **Text-to-Speech**: Panduan pengucapan
- **Speech Recognition**: Verifikasi dengan Whisper AI

## 📁 Struktur Folder

```
VoiceBoost/
├── app.py                          # Aplikasi utama Streamlit
├── requirements.txt                # Dependencies
├── train_dysarthria_model.py         
├── config.py 
├── README.md                       # Dokumentasi
├── models/                         # Model yang sudah dilatih
│   └── dysarthria_model.joblib
├── recordings/                     # Rekaman sementara user
├── data_training/                  # Dataset training
│   ├── dysarthria/
│   │   ├── M01/
│   │   │   ├── Session1/
│   │   │   │   ├── prompts/        # File transkrip (.txt)
│   │   │   │   ├── wav_arrayMic/   # Audio array mic (.wav)
│   │   │   │   └── wav_headMic/    # Audio head mic (.wav)
│   │   │   └── Session2/
│   │   └── M02/
│   └── normal/
│       ├── MC01/
│       └── MC02/
├── data_testing/                   # Dataset testing (struktur sama)

```

## 🛠️ Setup dan Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd VoiceBoost
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Dataset
Pastikan folder `data_training` dan `data_testing` sudah tersedia dengan struktur yang benar:
- Setiap kategori (dysarthria/normal) berisi folder partisipan (M01, M02, MC01, etc.)
- Setiap partisipan berisi folder sesi (Session1, Session2, etc.)
- Setiap sesi berisi:
  - `prompts/`: File transkrip (.txt)
  - `wav_arrayMic/` atau `wav_headMic/`: File audio (.wav)

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```

## 📊 Parameter Analisis

Sistem menganalisis 13+ fitur suara untuk klasifikasi dysarthria:

### Fitur Utama:
1. **Formant Frequencies (F1, F2, F3)** - Kejelasan vokal
2. **Vowel Space Area (VSA)** - Ruang artikulasi vokal  
3. **Speaking Rate** - Kecepatan bicara (kata/menit)
4. **Intensity Variation** - Variasi volume suara
5. **Pitch Variation** - Variasi nada (F0 range)
6. **Jitter & Shimmer** - Stabilitas suara
7. **Pause Ratio** - Proporsi jeda dalam bicara

### Fitur Tambahan:
8. **Spectral Centroid** - Pusat spektral
9. **Spectral Rolloff** - Rolloff spektral
10. **Zero Crossing Rate** - Tingkat zero crossing
11. **Consonant-Vowel Ratio** - Rasio konsonan-vokal
12. **MFCC Features** - Mel-frequency cepstral coefficients

## 🎯 Cara Penggunaan

### 1. Training Model
1. Buka halaman **Home**
2. Masukkan path ke folder `data_training` dan `data_testing`
3. Klik **Train Model**
4. Tunggu proses training selesai
5. Model akan disimpan otomatis

### 2. Speech Training
1. Pilih halaman **Speech Training**
2. Pilih mode: **Word** atau **Sentence**
3. Klik **Generate Random** untuk mendapat teks baru
4. Klik **Play Pronunciation** untuk mendengar contoh
5. Klik **Start Recording** untuk merekam suara
6. Klik **Analyze Speech** untuk mendapat hasil

### 3. Interpretasi Hasil
- **Green (Normal)**: Pola bicara normal
- **Red (Dysarthria)**: Terdeteksi gangguan bicara
- **Confidence Score**: Tingkat kepercayaan prediksi
- **Feature Analysis**: Analisis detail setiap parameter
- **Recommendations**: Saran berdasarkan hasil

## 🔧 Advanced Usage

### Validasi Dataset
```python
from utils.data_utils import validate_dataset_structure, print_dataset_summary

# Validasi struktur dataset
stats = validate_dataset_structure("data_training")
print_dataset_summary(stats)
```

### Ekstraksi Fitur Manual
```python
from app import AudioFeatureExtractor

extractor = AudioFeatureExtractor()
features = extractor.extract_all_features("path/to/audio.wav", "transcript")
print(features)
```

### Export ke CSV
```python
from utils.data_utils import create_dataset_manifest, export_features_to_csv

# Buat manifest dataset
create_dataset_manifest(["data_training", "data_testing"], "manifest.json")

# Export fitur ke CSV
export_features_to_csv("manifest.json", extractor, "features.csv")
```

## 🎛️ Konfigurasi

### Audio Settings
- **Sample Rate**: 22050 Hz (default)
- **Recording Duration**: 3-10 detik
- **Audio Format**: WAV, 16-bit

### Model Settings
- **Algorithm**: Random Forest (100 trees)
- **Features**: 13 fitur akustik
- **Normalization**: StandardScaler
- **Cross-validation**: Stratified split

### Thresholds
- **Silence Trimming**: -20 dB
- **Pause Detection**: 10% of mean RMS
- **Speaking Rate**: 60-300 WPM

## 🐛 Troubleshooting

### Error: "No module named 'librosa'"
```bash
pip install librosa soundfile
```

### Error: "CUDA not available"
- Normal, sistem akan menggunakan CPU
- Untuk GPU: install PyTorch dengan CUDA support

### Error: "Dataset not found"
- Pastikan folder `data_training` dan `data_testing` ada
- Periksa struktur folder sesuai dokumentasi

### Error: "Recording failed"
- Periksa permission microphone di browser
- Coba refresh halaman dan allow microphone access

### Performance Issues
- Kurangi jumlah file training untuk testing
- Gunakan audio dengan durasi < 10 detik
- Close aplikasi lain yang menggunakan microphone

## 📈 Model Performance

Dengan dataset yang cukup, sistem dapat mencapai:
- **Training Accuracy**: 85-95%
- **Test Accuracy**: 75-85%
- **Processing Time**: < 5 detik per audio
- **Supported Languages**: Bahasa Indonesia

## 🔬 Penelitian & Pengembangan

### Metode yang Digunakan:
- **Feature Extraction**: Librosa + custom algorithms
- **Classification**: Random Forest Classifier
- **Validation**: Speaker-independent split
- **Evaluation**: Precision, Recall, F1-score

### Future Improvements:
- Deep learning models (CNN, RNN)
- Multi-language support
- Real-time classification
- Mobile app version

## 📞 Support

Untuk pertanyaan atau bantuan:
1. Check troubleshooting section
2. Validate dataset structure dengan `utils/data_utils.py`
3. Check console log untuk error details
4. Ensure all dependencies installed correctly

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Librosa** - Audio analysis library
- **Streamlit** - Web app framework  
- **Scikit-learn** - Machine learning tools
- **OpenAI Whisper** - Speech recognition
- **gTTS** - Text-to-speech conversion