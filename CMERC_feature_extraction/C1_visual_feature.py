"""
C1_visual 조건 피처 추출 (CMERC / IEMOCAP inference-only)
=========================================================
- dropbox C0 pkl을 베이스로 로드 → Session5 utterance의 audio/visual만 교체
- audio: openSMILE IS10 Functionals(1582) → C0 raw-IS10 stats로 z-score
- visual: hsemotion enet_b0 pooled feature(frame-mean) → 342
- roberta: C0 것을 그대로 복사 (C1/C2는 텍스트 변경 없음)

★ C0/C2로 바꿀 때: 아래 ===설정=== 블록의 AVI_DIR / OUT_DIR / OUT_*_PKL 만 변경.
  단, C2는 stats 블록을 'raw IS10 직접 계산' 버전으로 교체해야 함 (이 파일은 C1: np.load 사용).

⚠️ 주의: 설정/모델로딩/메인루프/저장 = 대화 로그 verbatim.
         extract_audio() / extract_visual() 본문 = 재구성(RECONSTRUCTED), 검증 필요.
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
# ===설정=== (C0/C1/C2 전환 시 여기만 변경)
# ============================================================
# BASE_DIR = 이 스크립트가 있는 폴더(= MM_rebuttal/). MM_BASE_DIR 로 덮어쓸 수 있음.
BASE_DIR       = os.environ.get("MM_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
C0_RAW_PKL     = f"{BASE_DIR}/CMERC/dropbox_data/IEMOCAP_features.pkl"
C0_ROB_PKL     = f"{BASE_DIR}/CMERC/dropbox_data/iemocap_features_roberta.pkl"
AVI_DIR        = f"{BASE_DIR}/IEMOCAP_intervention/visual"             # ← C1 (시각개입 영상)
ORIG_AVI_DIR   = f"{BASE_DIR}/IEMOCAP_full_release/Session5/sentences/avi_flat"  # stats 계산용 원본
# audio 정규화 stats: C0 기준 raw IS10 통계 npy (C1이 쓰던 파일)
AUDIO_MEAN_NPY = f"{BASE_DIR}/extracted_features/iemocap_audio_orig_mean.npy"
AUDIO_STD_NPY  = f"{BASE_DIR}/extracted_features/iemocap_audio_orig_std.npy"
OUT_DIR        = f"{BASE_DIR}/cmerc_again/C1_visual"                   # ← C1
OUT_RAW_PKL    = f"{OUT_DIR}/IEMOCAP_features_C1.pkl"                  # ← C1
OUT_ROB_PKL    = f"{OUT_DIR}/iemocap_features_roberta_C1.pkl"         # ← C1
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

# hsemotion 내부 torch.load 패치 (weights_only 이슈 회피, 재실행 안전)
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
    feature_level=opensmile.FeatureLevel.Functionals,   # ★ Functionals → 1582 (CMERC 포맷)
)
print(f"IS10 dim: {len(smile.feature_names)}")

# ── audio 정규화 stats (npy 있으면 로드, 없으면 원본 Session5에서 per-feature 계산) ──
if os.path.exists(AUDIO_MEAN_NPY) and os.path.exists(AUDIO_STD_NPY):
    aud_mean = np.load(AUDIO_MEAN_NPY)
    aud_std  = np.load(AUDIO_STD_NPY)
    print(f"Audio norm stats: mean={aud_mean.mean():.4f}, std={aud_std.mean():.4f}")
else:
    print("stats npy 없음 → 원본 Session5에서 raw IS10 통계 계산 (최초 1회)...")
    raw_stats_list = []
    for fname in tqdm(sorted(os.listdir(ORIG_AVI_DIR)), desc="Computing stats"):
        if not fname.endswith('.avi'):
            continue
        avi_path = os.path.join(ORIG_AVI_DIR, fname)
        tmp_wav  = avi_path.replace('.avi', '_stat_tmp.wav')
        try:
            subprocess.run(['ffmpeg', '-y', '-i', avi_path, '-vn',
                            '-ar', '16000', '-ac', '1', tmp_wav],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raw_stats_list.append(smile.process_file(tmp_wav).values[0].astype(np.float32))
        except Exception:
            pass
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
    raw_arr  = np.stack(raw_stats_list)
    aud_mean = raw_arr.mean(axis=0)          # ★ per-feature (1582,)
    aud_std  = raw_arr.std(axis=0) + 1e-8    # ★ per-feature (1582,) — 스칼라로 하면 값이 폭발함
    os.makedirs(os.path.dirname(AUDIO_MEAN_NPY), exist_ok=True)
    np.save(AUDIO_MEAN_NPY, aud_mean)
    np.save(AUDIO_STD_NPY,  aud_std)
    print(f"Stats saved: mean={aud_mean.mean():.4f}, std={aud_std.mean():.4f}")

print("All models loaded!")


# ============================================================
# ⚠️ RECONSTRUCTED: 아래 두 함수 본문은 대화 로그가 잘려 추정 복원함.
#    extract_visual의 342차원 추출(enet pooled → [:VISUAL_DIM])이 가장 불확실.
# ============================================================
def extract_audio(avi_path):
    """avi → (1582,) z-scored IS10 Functionals   ⚠️ RECONSTRUCTED"""
    tmp_wav = avi_path.replace('.avi', '_tmp.wav')
    try:
        subprocess.run(['ffmpeg', '-y', '-i', avi_path, '-vn',
                        '-ar', '16000', '-ac', '1', tmp_wav],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        feat = smile.process_file(tmp_wav).values[0].astype(np.float32)  # (1582,)
        feat = (feat - aud_mean) / aud_std                               # C0 raw-IS10 기준 z-score
        return feat.astype(np.float32)
    except Exception:
        return np.zeros(AUDIO_DIM, dtype=np.float32)
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
            _ = enet(t)                                  # hook이 _pool_cache 채움
        pooled = _pool_cache['out'].cpu().numpy().flatten()
        feats.append(pooled[:VISUAL_DIM].astype(np.float32))   # ⚠️ 342 슬라이싱 (불확실)
    cap.release()
    return np.mean(feats, axis=0).astype(np.float32) if feats else np.zeros(VISUAL_DIM, dtype=np.float32)


# ── 메인 루프 (Session5만) ────────────────────────────────────
new_audio  = dict(c0_raw[4])
new_visual = dict(c0_raw[5])

ses5_vids = [vid for vid in all_vids if 'Ses05' in str(vid)]
print(f"Session5 dialogues: {len(ses5_vids)}")

for vid in tqdm(ses5_vids, desc="C1 feature extraction (Ses05 only)"):
    utt_ids = videoIDs[vid]
    a_list, v_list = [], []

    for i, uid in enumerate(utt_ids):
        uid_str  = str(uid) if not isinstance(uid, str) else uid
        avi_path = avi_index.get(uid_str)

        if avi_path and os.path.exists(avi_path):
            a_list.append(extract_audio(avi_path))
            v_list.append(extract_visual(avi_path))
        else:
            tqdm.write(f"  [WARN] {uid_str} not found, keeping C0 feat")
            a_list.append(np.array(c0_raw[4][vid][i], dtype=np.float32))
            v_list.append(np.array(c0_raw[5][vid][i], dtype=np.float32))

    new_audio[vid]  = np.stack(a_list).astype(np.float32)
    new_visual[vid] = np.stack(v_list).astype(np.float32)

# ── 저장 ──────────────────────────────────────────────────────
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