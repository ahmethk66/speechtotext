import subprocess
import torch
import whisper
import pyannote.audio
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
import numpy as np
from datetime import timedelta

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Modeli yükleyin
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

def extract_speakers(model, path, num_speakers=2):
    """Do diarization with speaker names"""
    
    mono = 'cagri.kaydi1_mono.wav'
    
    # FFmpeg ile ses dosyasını mono'ya dönüştürün
    cmd = f'ffmpeg -i {path} -af "pan=mono|c0=0.5*c0+0.5*c1" -y {mono}'
    subprocess.check_output(cmd, shell=True)
    
    result = model.transcribe(mono)
    segments = result["segments"]
    
    with contextlib.closing(wave.open(mono, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        
    audio = Audio()
    
    def segment_embedding(segment):
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(mono, clip)
        
        # Kontrol: Mono olup olmadığını doğrulayın
        assert waveform.shape[0] == 1, "Waveform is not mono"
        
        return embedding_model(waveform[None])

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)
    
    embeddings = np.nan_to_num(embeddings)
    
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
        
    return segments    

def write_segments(segments, outfile):
    """write out segments to file"""
    
    def time(secs):
        return timedelta(seconds=round(secs))
    
    with open(outfile, "w") as f:    
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
            f.write(segment["text"][1:] + ' ')

# Modeli yükleyin
model = whisper.load_model('medium')

# Dosya yolu
path = 'C:\\Users\\ahmet\\Desktop\\cagri.kayitlar\\cagri.kaydi1.wav'

# Konuşmacı çıkarımı yapın
seg = extract_speakers(model, path)

# Segmentleri dosyaya yazın
write_segments(seg, 'transcript2.txt')
