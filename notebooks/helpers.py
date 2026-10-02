import numpy as np
import librosa


FEATURE_NAMES = (
    [f"mfcc{i+1}_mean" for i in range(13)] +
    [f"mfcc{i+1}_std" for i in range(13)] +
    ["centroid_mean", "centroid_std"] +
    ["bandwidth_mean", "bandwidth_std"] +
    ["rolloff_mean", "rolloff_std"] +
    ["zcr_mean", "zcr_std"] +
    ["rms_mean", "rms_std"] +
    [f"chroma{i+1}_mean" for i in range(12)] +
    [f"chroma{i+1}_std" for i in range(12)]
)


def extract_features(path, sr=32000):
    y, sr = librosa.load(path, sr=sr)
    if len(y) == 0:
        return np.zeros(60)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        centroid.mean(axis=1), centroid.std(axis=1),
        bandwidth.mean(axis=1), bandwidth.std(axis=1),
        rolloff.mean(axis=1), rolloff.std(axis=1),
        zcr.mean(axis=1), zcr.std(axis=1),
        rms.mean(axis=1), rms.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
    ])
    return features


def _summarize_extended(feat_2d):
    """Given a 2D feature array (n_features, n_frames), return extended stats."""
    return np.concatenate([
        feat_2d.mean(axis=1),
        feat_2d.std(axis=1),
        feat_2d.min(axis=1),
        feat_2d.max(axis=1),
        np.percentile(feat_2d, 25, axis=1),
        np.percentile(feat_2d, 75, axis=1),
    ])


def extract_features_extended(path, sr=32000):
    y, sr = librosa.load(path, sr=sr)
    if len(y) == 0:
        return np.zeros(228)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    all_feats = np.vstack([mfcc, centroid, bandwidth, rolloff, zcr, rms, chroma])
    return _summarize_extended(all_feats)


# feature set column indices for SET_C (60 features)
# SET_A: MFCCs only (mean+std) = 26 features
SET_A_COLS = list(range(0, 26))
# SET_B: MFCCs + spectral (centroid, bandwidth, rolloff, zcr, rms) = 36 features
SET_B_COLS = list(range(0, 36))
# SET_C: all 60
SET_C_COLS = list(range(0, 60))


def extract_segment(path, start_sec, end_sec, sr=32000):
    duration = end_sec - start_sec
    y, sr = librosa.load(path, sr=sr, offset=start_sec, duration=duration)
    if len(y) == 0:
        return np.zeros(60)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        centroid.mean(axis=1), centroid.std(axis=1),
        bandwidth.mean(axis=1), bandwidth.std(axis=1),
        rolloff.mean(axis=1), rolloff.std(axis=1),
        zcr.mean(axis=1), zcr.std(axis=1),
        rms.mean(axis=1), rms.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
    ])
    return features
