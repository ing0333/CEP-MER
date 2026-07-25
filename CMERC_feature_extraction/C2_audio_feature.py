"""
C2_audio 조건 피처 추출 (CMERC / IEMOCAP inference-only)
=======================================================
★ 이 버전이 12% → ~60%로 잡힌 "작동한 C2"다.
  핵심: audio 정규화 stats를 dropbox/AUDIO_MEAN_NPY가 아니라
        '원본 Session5 AVI에서 raw IS10를 직접 계산(per-feature mean/std)'해서 사용.
  (per-feature axis=0 통계여야 함. 스칼라 std로 나누면 값이 폭발했던 게 12% 버전의 원인)

C1과 구조 동일, 차이는 AVI_DIR/OUT 경로 + stats 블록(raw IS10 직접 계산).

⚠️ 신뢰도:
   - 설정 / 모델로드 / stats블록 / 메인루프 / 저장 = 대화 로그 verbatim
   - extract_audio / extract_visual 본문 = 재구성(RECONSTRUCTED), 검증 필요
"""

import os, pickle, subprocess, warnings, shutil
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import opensmile

warnings.filterwarnings("ignore")

# ============================================================
# ===설정===  (verbatim)
# ============================================================
# BASE_DIR = 이 스크립트가 있는 폴더(= MM_rebuttal/). MM_BASE_DIR 로 덮어쓸 수 있음.
BASE_DIR       = os.environ.get("MM_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
C0_RAW_PKL     = f"{BASE_DIR}/CMERC/dropbox_data/IEMOCAP_features.pkl"
C0_ROB_PKL     = f"{BASE_DIR}/CMERC/dropbox_data/iemocap_features_roberta.pkl"
AVI_DIR        = f"{BASE_DIR}/IEMOCAP_intervention/audio"                    # ← 음성개입 영상
ORIG_AVI_DIR   = f"{BASE_DIR}/IEMOCAP_full_release/Session5/sentences/avi_flat"  # stats 계산용 원본
STATS_MEAN_NPY = f"{BASE_DIR}/cmerc_again/iemocap_raw_is10_mean.npy"
STATS_STD_NPY  = f"{BASE_DIR}/cmerc_again/iemocap_raw_is10_std.npy"
OUT_DIR        = f"{BASE_DIR}/cmerc_again/C2_audio"
OUT_RAW_PKL    = f"{OUT_DIR}/IEMOCAP_features_C2.pkl"
OUT_ROB_PKL    = f"{OUT_DIR}/iemocap_features_roberta_C2.pkl"
# ============================================================

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
VISUAL_DIM = 342
AUDIO_DIM  = 1582
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading C0 pkls...")
with open(C0_RAW_PKL, 'rb') as f:
    c0_raw = list(pickle.load(f, encoding='latin1'))

videoIDs = c0_raw[0]
all_vids = list(videoIDs.keys())
print(f"Total dialogues: {len(all_vids)}")

print(f"Indexing avi from: {AVI_DIR}")
avi_index = {}
for fname in os.listdir(AVI_DIR):
    if fname.endswith('.avi'):
        avi_index[fname.replace('.avi', '')] = os.path.join(AVI_DIR, fname)
print(f"Found {len(avi_index)} avi files")

print("Loading models...")

if not getattr(torch, '_hsemotion_patched', False):
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw['weights_only'] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load
    torch._hsemotion_patched = True

from hsemotion.facial_emotions import HSEmotionRecognizer
recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=DEVICE)
enet = recognizer.model.eval()

_pool_cache = {}
def _hook(m, i, o): _pool_cache['out'] = o.flatten(1)
recognizer.model.global_pool.register_forward_hook(_hook)

img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals,    # ★ 1582 (CMERC 포맷)
)
print(f"IS10 dim: {len(smile.feature_names)}")

# ── ★ 핵심: Raw IS10 stats (없으면 원본 Session5에서 per-feature 계산 → npy 캐시) ──
if os.path.exists(STATS_MEAN_NPY) and os.path.exists(STATS_STD_NPY):
    aud_mean = np.load(STATS_MEAN_NPY)
    aud_std  = np.load(STATS_STD_NPY)
    print(f"Loaded cached stats: mean={aud_mean.mean():.4f}, std={aud_std.mean():.4f}")
else:
    print("Computing raw IS10 stats from original Session5 (first time only)...")
    raw_stats_list = []
    for fname in tqdm(sorted(os.listdir(ORIG_AVI_DIR)), desc="Computing stats"):
        if not fname.endswith('.avi'):
            continue
        avi_path = os.path.join(ORIG_AVI_DIR, fname)
        tmp_wav  = avi_path.replace('.avi', '_stat_tmp.wav')
        try:
            # '-vn' 은 extract_audio() 와 동일하게 유지해야 stats/추출 경로가 일치한다.
            subprocess.run(['ffmpeg', '-y', '-i', avi_path, '-vn',
                            '-ar', '16000', '-ac', '1', tmp_wav],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            feat = smile.process_file(tmp_wav).values[0].astype(np.float32)
            raw_stats_list.append(feat)
        except:
            pass
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
    raw_arr  = np.stack(raw_stats_list)
    aud_mean = raw_arr.mean(axis=0)          # ★ per-feature (1582,) — 스칼라로 하면 안 됨
    aud_std  = raw_arr.std(axis=0) + 1e-8    # ★ per-feature (1582,)
    np.save(STATS_MEAN_NPY, aud_mean)
    np.save(STATS_STD_NPY,  aud_std)
    print(f"Stats saved: mean={aud_mean.mean():.4f}, std={aud_std.mean():.4f}")  # ~27.8 / ~27.9

print("All models loaded!")


# ============================================================
# ⚠️ RECONSTRUCTED: 아래 두 함수 본문은 추정 복원. 검증 필요.
# ============================================================
def extract_audio_raw(avi_path):
    """avi → (1582,) RAW IS10 Functionals (정규화 전)."""
    tmp_wav = avi_path.replace('.avi', '_tmp.wav')
    try:
        subprocess.run(['ffmpeg', '-y', '-i', avi_path, '-vn',
                        '-ar', '16000', '-ac', '1', tmp_wav],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return smile.process_file(tmp_wav).values[0].astype(np.float32)   # (1582,)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def extract_visual(avi_path):
    """avi → (342,) hsemotion enet pooled feature (frame-mean)   ⚠️ RECONSTRUCTED"""
    cap = cv2.VideoCapture(avi_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros(VISUAL_DIM, dtype=np.float32)
    n_sample = min(8, total)
    indices = np.linspace(0, total - 1, n_sample, dtype=int)
    feats = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = img_tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _ = enet(t)
        pooled = _pool_cache['out'].cpu().numpy().flatten()
        feats.append(pooled[:VISUAL_DIM].astype(np.float32))   # ⚠️ 342 슬라이싱 (불확실)
    cap.release()
    return np.mean(feats, axis=0).astype(np.float32) if feats else np.zeros(VISUAL_DIM, dtype=np.float32)


# ── 메인 루프 (Session5만) ──  (verbatim)
new_audio  = dict(c0_raw[4])
new_visual = dict(c0_raw[5])

ses5_vids = [vid for vid in all_vids if 'Ses05' in str(vid)]
print(f"Session5 dialogues: {len(ses5_vids)}")

# 1단계: raw IS10 / visual 을 먼저 전부 모은다 (정규화는 아직 안 함).
raw_audio  = {}   # vid -> list of (1582,) raw  또는 None(원본 avi 없음 → C0 유지)
for vid in tqdm(ses5_vids, desc="C2 feature extraction (Ses05 only)"):
    utt_ids = videoIDs[vid]
    a_list, v_list = [], []

    for i, uid in enumerate(utt_ids):
        uid_str  = str(uid) if not isinstance(uid, str) else uid
        avi_path = avi_index.get(uid_str)

        if avi_path and os.path.exists(avi_path):
            a_list.append(extract_audio_raw(avi_path))     # None 가능 (추출 실패)
            v_list.append(extract_visual(avi_path))
        else:
            tqdm.write(f"  [WARN] {uid_str} not found, keeping C0 feat")
            a_list.append(None)
            v_list.append(np.array(c0_raw[5][vid][i], dtype=np.float32))

    raw_audio[vid]  = a_list
    new_visual[vid] = np.stack(v_list).astype(np.float32)

# ── 2단계: audio 정규화 ────────────────────────────────────────
# C2_AUDIO_NORM=self (기본) : C2 개입 오디오 자체의 per-feature 통계로 z-score
#              =orig        : 원본 Session5 통계로 z-score (STATS_*_NPY)
#
# 왜 self 가 기본인가:
#   VAD 마스킹으로 디지털 무음이 섞이면 IS10 functionals 분포가 원본과 크게
#   달라진다. 원본 통계로 z-score 하면 |z|>5 인 값이 9.4%까지 튀고(C0는 0.23%)
#   전체 std 가 4.79 가 되어 체크포인트가 학습한 입력 범위를 완전히 벗어난다.
#   -> inference 정확도 36% 로 붕괴.
#   CMERC 배포 pkl 자체도 "그 데이터셋의 통계로 z-score" 된 것이므로,
#   featurize 대상과 통계 대상을 일치시키는 self 쪽이 원 전처리와 일관적이다.
NORM_MODE = os.environ.get("C2_AUDIO_NORM", "self").lower()

flat = [f for v in ses5_vids for f in raw_audio[v] if f is not None]
if NORM_MODE == "self":
    arr      = np.stack(flat)
    aud_mean = arr.mean(axis=0)
    aud_std  = arr.std(axis=0) + 1e-8
    print(f"[norm] self  : mean={aud_mean.mean():.4f} std={aud_std.mean():.4f} (n={len(flat)})")
else:
    print(f"[norm] orig  : mean={aud_mean.mean():.4f} std={aud_std.mean():.4f}")

for vid in ses5_vids:
    out = []
    for i, f in enumerate(raw_audio[vid]):
        if f is None:
            out.append(np.array(c0_raw[4][vid][i], dtype=np.float32))
        else:
            out.append(((f - aud_mean) / aud_std).astype(np.float32))
    new_audio[vid] = np.stack(out).astype(np.float32)

# ── 저장 ──  (verbatim)
print("\nSaving...")
c0_raw[4] = new_audio
c0_raw[5] = new_visual

with open(OUT_RAW_PKL, 'wb') as f:
    pickle.dump(tuple(c0_raw), f)
print(f"✅ {OUT_RAW_PKL}")

shutil.copy(C0_ROB_PKL, OUT_ROB_PKL)
print(f"✅ {OUT_ROB_PKL} (C0 roberta 복사)")

all_aud = np.vstack([new_audio[v] for v in ses5_vids])
all_vis = np.vstack([new_visual[v] for v in ses5_vids])
print(f"\n[Validation - Session5]")
print(f"Audio  mean={all_aud.mean():.4f} std={all_aud.std():.4f}")   # ≈0 / ≈1 나와야 정상
print(f"Visual mean={all_vis.mean():.4f} nonzero={(all_vis!=0).mean():.4f}")