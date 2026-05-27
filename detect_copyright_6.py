#!/usr/bin/env python3
"""
detect_copyright_6.py  —  두 단계 탐지 (후보 수집 + 오프셋 일관성 검증)

  1단계: MFCC+Chroma / 창별 z-score 정규화 / 낮은 후보 임계값
         → (의심본 창, 원본 창, 유사도) 전체 매핑 수집
  2단계: offset = susp_time − orig_time 이 일정한 묶음만 채택
         · 진짜 사용: 연속된 창이 모두 같은 offset → 클러스터 형성
         · 우연 일치: offset 제각각 → 클러스터 미형성 → 자동 탈락
         + 단조성 검증 (susp↑ orig↑ 동반) + 연속성 검증 (창 간격 ≤ 2s)
"""

import sys, subprocess, os, platform

# ── 자동 설치 ─────────────────────────────────────────────────────

def run_cmd(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *packages])

def check_import(mod):
    import importlib
    try:   importlib.import_module(mod); return True
    except ImportError: return False

def install_python_deps():
    mapping = {"librosa":"librosa","numpy":"numpy","scipy":"scipy",
                "Pillow":"PIL","imagehash":"imagehash"}
    missing = [p for p,m in mapping.items() if not check_import(m)]
    if missing:
        print(f"[설치 중] {', '.join(missing)}")
        pip_install(*missing)
        print("[완료] Python 패키지 설치")

def ffmpeg_exists():
    return run_cmd(["ffmpeg", "-version"]).returncode == 0

def install_ffmpeg():
    if ffmpeg_exists(): return True
    osn = platform.system()
    print("[설치 중] ffmpeg ...")
    if osn == "Darwin":
        if run_cmd(["which","brew"]).returncode != 0:
            os.system('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        if run_cmd(["brew","install","ffmpeg"]).returncode != 0: return False
    elif osn == "Windows":
        if run_cmd(["winget","install","--id","Gyan.FFmpeg","-e","--silent"]).returncode != 0:
            if run_cmd(["choco","install","ffmpeg","-y"]).returncode != 0: return False
        for p in [r"C:\Program Files\ffmpeg\bin", r"C:\ProgramData\chocolatey\bin"]:
            if os.path.exists(p) and p not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + p
    elif osn == "Linux":
        if run_cmd(["sudo","apt-get","install","-y","ffmpeg"]).returncode != 0: return False
    else: return False
    ok = ffmpeg_exists()
    if ok: print("[완료] ffmpeg 설치")
    return ok

print("[시작] 환경 확인 중...")
install_python_deps()
ffmpeg_ok = install_ffmpeg()
print("[준비 완료] GUI 시작\n")

import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
import threading, tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import librosa

# ── 핵심 상수 ─────────────────────────────────────────────────────
AUDIO_SR         = 11025   # 22050의 절반 → 메모리 절반, 정확도 동일
HOP_LENGTH       = 256    # 시간 해상도 유지 (256/11025 ≈ 512/22050)
N_MFCC           = 20
N_CHROMA         = 12

WINDOW_SEC       = 1.0    # 1초 창 → 2초 클립도 탐지 가능
STEP_SEC         = 0.25   # 0.25초 간격 (촘촘)
CAND_THRESH      = 0.42   # 내부 후보 임계값 (낮음 — 일관성 필터가 거름)
OFFSET_BIN_S     = 1.0    # 오프셋 허용 오차 (초)
MIN_HITS         = 4      # 클러스터 최소 창 수 (4×0.25s = 1초 이상 연속)
MIN_SUSP_SPAN_S  = 0.75   # 의심본 창 범위 최솟값
MAX_GAP_S        = 2.0    # 클러스터 내 최대 창 간격
MERGE_GAP_S      = 2.0    # 인접 구간 병합 허용 거리
MIN_SEGMENT_S    = 1.0    # 최종 표시 최소 구간
FRAME_SAMPLE_RATE = 1
# ─────────────────────────────────────────────────────────────────


def fmt_time(s: float) -> str:
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}.{int((s-int(s))*100):02d}"


def extract_audio(video_path, out_wav):
    r = subprocess.run(
        ["ffmpeg","-y","-i",video_path,"-vn","-ac","1",
         "-ar",str(AUDIO_SR),"-f","wav",out_wav,"-loglevel","error"],
        capture_output=True)
    return r.returncode == 0


def extract_features(wav_path):
    """MFCC(20) + Chroma(12) — 5분 단위 청크로 처리하여 OOM 방지."""
    y, sr = librosa.load(wav_path, sr=AUDIO_SR, mono=True)

    CHUNK_SAMP = 5 * 60 * sr   # 5분 = 6,615,000 샘플
    mfcc_parts, chroma_parts = [], []

    for start in range(0, len(y), CHUNK_SAMP):
        chunk = y[start : start + CHUNK_SAMP]
        if len(chunk) < HOP_LENGTH * 4:
            break
        mfcc_parts.append(
            librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH))
        chroma_parts.append(
            librosa.feature.chroma_stft(y=chunk, sr=sr,
                                         hop_length=HOP_LENGTH,
                                         n_chroma=N_CHROMA))

    if not mfcc_parts:
        feat = np.zeros((N_MFCC + N_CHROMA, 1), dtype=np.float32)
        return feat, float(HOP_LENGTH / sr)

    feat = np.vstack([
        np.hstack(mfcc_parts),
        np.hstack(chroma_parts),
    ]).astype(np.float32)
    return feat, float(HOP_LENGTH / sr)


def _zn(w: np.ndarray) -> np.ndarray:
    """창 단위 z-score — 볼륨/믹싱/EQ 차이를 상쇄."""
    mu  = w.mean(axis=1, keepdims=True)
    std = w.std(axis=1, keepdims=True) + 1e-8
    return (w - mu) / std


def find_segments(feat_orig, feat_susp, hop_dur, user_thresh, progress_cb=None):
    """
    1단계: 후보 수집 (낮은 임계값, 벡터화 NCC)
    2단계: 오프셋 일관성 필터 → 진짜 사용 구간만 추출
    """
    W    = max(1, int(WINDOW_SEC   / hop_dur))
    S    = max(1, int(STEP_SEC     / hop_dur))
    OBIN = max(1, int(OFFSET_BIN_S / hop_dur))
    F    = feat_orig.shape[0]
    No, Ns = feat_orig.shape[1], feat_susp.shape[1]

    if No < W or Ns < W:
        if progress_cb: progress_cb(100)
        return []

    # orig_mat 예상 메모리 초과 시 스텝 자동 증가 (최대 400 MB)
    MAX_MAT_MB = 400
    while True:
        est_mb = (((No - W) // S + 1) * F * W * 4) / (1024 ** 2)
        if est_mb <= MAX_MAT_MB or S >= W:
            break
        S *= 2
    OBIN = max(1, int(OFFSET_BIN_S / hop_dur))

    # ── 1단계: 원본 창 행렬 사전 계산 ────────────────────────────
    orig_pos = list(range(0, No - W + 1, S))
    orig_mat = np.zeros((len(orig_pos), F * W), dtype=np.float32)
    for i, op in enumerate(orig_pos):
        orig_mat[i] = _zn(feat_orig[:, op:op+W]).flatten()
    orig_norms = np.linalg.norm(orig_mat, axis=1).astype(np.float32)

    susp_pos = list(range(0, Ns - W + 1, S))
    total    = max(1, len(susp_pos))

    # 후보 저장 리스트
    sp_list, op_list, sim_list = [], [], []

    for i, sp in enumerate(susp_pos):
        if progress_cb and i % 60 == 0:
            progress_cb(int(78 * i / total))

        ws = _zn(feat_susp[:, sp:sp+W]).flatten().astype(np.float32)
        wn = float(np.linalg.norm(ws))
        if wn < 1e-8:
            continue

        dots   = orig_mat @ ws                          # (N_orig,)
        denoms = orig_norms * wn
        sims   = np.where(denoms > 1e-8, dots / denoms, 0.0)

        good = np.where(sims >= CAND_THRESH)[0]
        # 창당 최대 15개만 유지 (메모리 제한)
        if len(good) > 15:
            good = good[np.argsort(sims[good])[-15:]]
        for idx in good:
            sp_list.append(sp)
            op_list.append(orig_pos[idx])
            sim_list.append(float(sims[idx]))

    if progress_cb: progress_cb(82)

    if not sp_list:
        if progress_cb: progress_cb(100)
        return []

    sp_arr  = np.array(sp_list,  dtype=np.int32)
    op_arr  = np.array(op_list,  dtype=np.int32)
    sim_arr = np.array(sim_list, dtype=np.float32)
    off_arr = (sp_arr - op_arr).astype(np.int32)

    # ── 2단계: 오프셋 일관성 클러스터링 ─────────────────────────
    off_min, off_max = int(off_arr.min()), int(off_arr.max())
    n_bins = max(1, (off_max - off_min) // OBIN + 1)
    hist, edges = np.histogram(off_arr, bins=n_bins,
                                range=(off_min - 0.5, off_max + 0.5))

    if progress_cb: progress_cb(88)

    segs = []
    for b in np.where(hist >= MIN_HITS)[0]:
        lo, hi = edges[b], edges[b + 1]
        mask = (off_arr >= lo) & (off_arr < hi)

        sp_b   = sp_arr[mask]
        op_b   = op_arr[mask]
        sim_b  = sim_arr[mask]

        # ① 고유 suspect 위치 수 / 범위
        u_sp = np.unique(sp_b)
        if len(u_sp) < MIN_HITS:
            continue
        span_s = float((u_sp.max() - u_sp.min()) * hop_dur)
        if span_s < MIN_SUSP_SPAN_S:
            continue

        # ② 창 간격 연속성 (듬성듬성이면 우연 일치)
        if len(u_sp) > 1:
            max_gap = float(np.diff(np.sort(u_sp)).max() * hop_dur)
            if max_gap > MAX_GAP_S:
                continue

        # ③ 단조성: susp↑ orig↑ 함께 증가해야 함
        if len(u_sp) >= 4:
            best_op_per_sp = {}
            for sp, op, sim in zip(sp_b.tolist(), op_b.tolist(), sim_b.tolist()):
                if sp not in best_op_per_sp or sim > best_op_per_sp[sp][1]:
                    best_op_per_sp[sp] = (op, sim)
            u_sp_s = np.sort(u_sp)
            u_op_s = np.array([best_op_per_sp[s][0] for s in u_sp_s], dtype=np.float32)
            if u_sp_s.std() > 0 and u_op_s.std() > 0:
                corr = float(np.corrcoef(u_sp_s.astype(np.float32), u_op_s)[0, 1])
                if np.isnan(corr) or corr < 0.50:
                    continue

        # ④ 유저 임계값 최종 필터
        best_sim = float(sim_b.max())
        if best_sim < user_thresh:
            continue

        segs.append({
            "susp_start":  float(sp_b.min() * hop_dur),
            "susp_end":    float((sp_b.max() + W) * hop_dur),
            "orig_start":  float(op_b.min() * hop_dur),
            "orig_end":    float((op_b.max() + W) * hop_dur),
            "similarity":  round(best_sim, 4),
            "hit_count":   int(len(u_sp)),
        })

    if progress_cb: progress_cb(100)
    segs.sort(key=lambda s: s["susp_start"])
    return _merge(segs)


def _merge(segs):
    if not segs:
        return []
    merged, cur = [], dict(segs[0])
    for h in segs[1:]:
        if h["susp_start"] - cur["susp_end"] <= MERGE_GAP_S:
            cur["susp_end"]   = max(cur["susp_end"],   h["susp_end"])
            cur["orig_end"]   = max(cur["orig_end"],   h["orig_end"])
            cur["similarity"] = max(cur["similarity"], h["similarity"])
            cur["hit_count"]  = cur.get("hit_count", 0) + h.get("hit_count", 0)
        else:
            merged.append(cur)
            cur = dict(h)
    merged.append(cur)
    return [s for s in merged if s["susp_end"] - s["susp_start"] >= MIN_SEGMENT_S]


# ── 영상 검증 ─────────────────────────────────────────────────────

def extract_frames(video_path, start, end, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    pat = os.path.join(out_dir, f"{prefix}_%04d.png")
    subprocess.run(["ffmpeg","-y","-ss",str(start),"-to",str(end),
                    "-i",video_path,"-vf",f"fps={FRAME_SAMPLE_RATE}",
                    pat,"-loglevel","error"], capture_output=True)
    return sorted(Path(out_dir).glob(f"{prefix}_*.png"))

def phash_sim(p1, p2):
    try:
        import imagehash; from PIL import Image
        h1 = imagehash.phash(Image.open(p1))
        h2 = imagehash.phash(Image.open(p2))
        return round(1.0 - (h1 - h2) / 64.0, 4)
    except Exception:
        return -1.0

def verify_video(orig_path, susp_path, seg, tmp_dir):
    of = extract_frames(orig_path, seg["orig_start"], seg["orig_end"],
                        os.path.join(tmp_dir, "orig"), "orig")
    sf = extract_frames(susp_path, seg["susp_start"], seg["susp_end"],
                        os.path.join(tmp_dir, "susp"), "susp")
    if not of or not sf: return -1.0
    n = min(len(of), len(sf))
    sims = [phash_sim(str(of[i]), str(sf[i])) for i in range(n)]
    return -1.0 if -1.0 in sims else round(float(np.mean(sims)), 4)


# ── 결과 저장 ─────────────────────────────────────────────────────

def save_result_txt(segments, orig_path, susp_path, thresh, save_dir) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(save_dir, f"result_{ts}.txt")
    L = []
    L.append("=" * 70)
    L.append("  저작권 침해 의심 구간 분석 결과  (v6 — 오프셋 일관성 탐지)")
    L.append(f"  분석 일시  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"  원본       : {orig_path}")
    L.append(f"  비교본     : {susp_path}")
    L.append(f"  유사도 임계: {thresh:.2f}  |  방식: MFCC+Chroma + 오프셋 일관성 검증")
    L.append("=" * 70)
    if not segments:
        L.append("  침해 의심 구간 없음")
    else:
        L.append(f"  총 {len(segments)}개 구간 발견")
        L.append("")
        L.append(f"  {'#':>3}  {'비교본 구간':^28}  {'원본 매칭 구간':^28}  음성   영상  창수")
        L.append("  " + "-" * 72)
        for i, s in enumerate(segments, 1):
            sr_ = f"{fmt_time(s['susp_start'])} ~ {fmt_time(s['susp_end'])}"
            or_ = f"{fmt_time(s['orig_start'])} ~ {fmt_time(s['orig_end'])}"
            vs  = f"{s['video_similarity']:.2f}" if s.get("video_similarity",-1) >= 0 else " N/A"
            hc  = str(s.get("hit_count", "?"))
            L.append(f"  {i:>3}  {sr_:^28}  {or_:^28}  {s['similarity']:.2f}  {vs:>4}  {hc}")
        L.append("")
        L.append("  [구간 상세]")
        for i, s in enumerate(segments, 1):
            dur = s["susp_end"] - s["susp_start"]
            vs_str = (f"  영상 유사도: {s['video_similarity']:.4f}"
                      if s.get("video_similarity",-1) >= 0 else "")
            L.append(f"  구간 {i}: 비교본  {fmt_time(s['susp_start'])} ~ "
                     f"{fmt_time(s['susp_end'])}  ({dur:.1f}초)")
            L.append(f"          원본     {fmt_time(s['orig_start'])} ~ "
                     f"{fmt_time(s['orig_end'])}")
            L.append(f"          음성 유사도: {s['similarity']:.4f}  "
                     f"일치 창 수: {s.get('hit_count','?')}{vs_str}")
            L.append("")
    L.append("=" * 70)
    L.append("[참고] 오프셋 일관성 탐지: 우연한 유사도는 자동 제거됩니다.")
    L.append("       이 결과는 1차 스크리닝용이며 사람의 직접 확인 필요")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return out


# ── GUI ──────────────────────────────────────────────────────────

class PairRow:
    """비교 쌍 한 행 (원본 + 비교본 + 진행바 + 상태)"""

    def __init__(self, container, remove_cb, idx):
        self.orig_var = tk.StringVar()
        self.susp_var = tk.StringVar()
        self._stvar   = tk.StringVar(value="⏳ 대기")

        bg = "#ffffff"
        self.frame = tk.Frame(container, bg=bg, relief="groove", bd=1)
        self.frame.pack(fill="x", padx=4, pady=2)

        tk.Label(self.frame, text=f"{idx:02d}", width=3,
                 bg=bg, fg="#aaa", font=("", 9)).pack(side="left", padx=(4, 0))

        # 원본
        tk.Entry(self.frame, textvariable=self.orig_var, width=22,
                 state="readonly", font=("", 9)).pack(side="left", padx=(4, 2))
        tk.Button(self.frame, text="저작권 영상", width=8, font=("", 9),
                  command=lambda: self._pick(self.orig_var)).pack(side="left", padx=(0, 8))

        # 비교본
        tk.Entry(self.frame, textvariable=self.susp_var, width=22,
                 state="readonly", font=("", 9)).pack(side="left", padx=(0, 2))
        tk.Button(self.frame, text="의심 영상", width=7, font=("", 9),
                  command=lambda: self._pick(self.susp_var)).pack(side="left", padx=(0, 8))

        # 진행바
        self.bar = ttk.Progressbar(self.frame, length=120, mode="determinate")
        self.bar.pack(side="left", padx=(0, 6))

        # 상태
        self._lbl = tk.Label(self.frame, textvariable=self._stvar,
                              width=11, font=("", 9), bg=bg, fg="#888", anchor="w")
        self._lbl.pack(side="left", padx=(0, 4))

        # 삭제
        tk.Button(self.frame, text="✕", fg="#cc3333", bg=bg,
                  relief="flat", font=("", 10, "bold"),
                  command=lambda: remove_cb(self)).pack(side="right", padx=4)

    def _pick(self, var):
        p = filedialog.askopenfilename(
            filetypes=[("영상/오디오", "*.mp4 *.mov *.avi *.mkv *.mp3 *.wav"),
                       ("모든 파일", "*.*")])
        if p:
            var.set(p)

    def set_status(self, text, color="#888"):
        self._stvar.set(text)
        self._lbl.config(fg=color)

    def set_prog(self, v):
        self.bar["value"] = v

    def reset(self):
        self._stvar.set("⏳ 대기")
        self._lbl.config(fg="#888")
        self.bar["value"] = 0


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("저작권 침해 구간 탐지기  v6  — 다중 쌍 동시 분석")
        self.configure(bg="#f0f0f0")
        self.resizable(True, True)
        self.minsize(820, 660)
        self.pair_rows: list = []
        self._build_ui()
        self._add_pair()   # 기본 1행
        if not ffmpeg_ok:
            msg = {"Darwin":  "brew install ffmpeg",
                   "Windows": "winget install Gyan.FFmpeg",
                   "Linux":   "sudo apt install ffmpeg"
                   }.get(platform.system(), "https://ffmpeg.org")
            messagebox.showwarning("ffmpeg 필요",
                f"ffmpeg를 찾을 수 없습니다.\n\n{msg}\n\n설치 후 재실행하세요.")

    # ── UI 구성 ────────────────────────────────────────────────────
    def _build_ui(self):

        # ① 쌍 목록
        pairs_lf = tk.LabelFrame(self, text="  비교 쌍 목록  ",
                                  bg="#f0f0f0", padx=8, pady=6)
        pairs_lf.pack(fill="x", padx=14, pady=(14, 4))

        hdr = tk.Frame(pairs_lf, bg="#f0f0f0")
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(hdr,
                 text="  #    원본  (저작권 가진 영상)                  "
                      "비교본  (도용 의심 영상)               진행        상태",
                 font=("", 8), bg="#f0f0f0", fg="#999").pack(side="left")
        tk.Button(hdr, text="＋  쌍 추가",
                  command=self._add_pair,
                  bg="#16a34a", fg="white",
                  font=("", 9, "bold"), relief="flat",
                  padx=8, pady=2, cursor="hand2").pack(side="right")

        # 스크롤 가능한 행 목록
        outer = tk.Frame(pairs_lf, bg="#f0f0f0")
        outer.pack(fill="x")
        self._canvas = tk.Canvas(outer, bg="#f0f0f0", highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._list_frame = tk.Frame(self._canvas, bg="#f0f0f0")
        self._cwin = self._canvas.create_window(
            (0, 0), window=self._list_frame, anchor="nw")

        def _sync(e=None):
            bb = self._canvas.bbox("all")
            if bb:
                self._canvas.configure(
                    scrollregion=bb,
                    height=min(190, self._list_frame.winfo_reqheight() + 6))
            self._canvas.itemconfig(self._cwin,
                                     width=self._canvas.winfo_width())

        self._list_frame.bind("<Configure>", _sync)
        self._canvas.bind("<Configure>", _sync)
        self._canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # ② 옵션
        opt = tk.LabelFrame(self, text="  옵션  ", bg="#f0f0f0", padx=10, pady=6)
        opt.pack(fill="x", padx=14, pady=4)

        tk.Label(opt, text="유사도 임계값", bg="#f0f0f0").grid(
            row=0, column=0, sticky="w")
        self.thresh = tk.DoubleVar(value=0.45)
        tk.Scale(opt, from_=0.35, to=0.85, resolution=0.05,
                 orient="horizontal", variable=self.thresh,
                 length=200, bg="#f0f0f0", highlightthickness=0
                 ).grid(row=0, column=1, padx=6, sticky="w")
        self._tlbl = tk.Label(opt, text="0.45", width=4, bg="#f0f0f0")
        self._tlbl.grid(row=0, column=2, sticky="w")
        self.thresh.trace_add("write",
            lambda *_: self._tlbl.config(text=f"{self.thresh.get():.2f}"))
        tk.Label(opt,
                 text="← 낮을수록 민감  |  0.45 권장",
                 bg="#f0f0f0", fg="#777", font=("", 8)
                 ).grid(row=0, column=3, sticky="w", padx=6)

        self.do_video = tk.BooleanVar(value=False)
        tk.Checkbutton(
            opt,
            text="영상 2차 검증 포함  (느림 — 다중 쌍 동시 분석 시 권장 해제)",
            variable=self.do_video, bg="#f0f0f0"
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # ③ 실행
        run_frm = tk.Frame(self, bg="#f0f0f0")
        run_frm.pack(fill="x", padx=14, pady=6)
        self.run_btn = tk.Button(
            run_frm, text="▶  전체 동시 분석 시작",
            command=self._start,
            bg="#2563eb", fg="white", font=("", 11, "bold"),
            relief="flat", cursor="hand2", padx=14, pady=6)
        self.run_btn.pack(side="left")
        self.status = tk.StringVar(value="쌍을 추가하고 파일을 선택하세요")
        tk.Label(run_frm, textvariable=self.status,
                 bg="#f0f0f0", fg="#444", font=("", 10)
                 ).pack(side="left", padx=14)

        # ④ 결과 로그
        res = tk.LabelFrame(self, text="  결과 로그  ",
                             bg="#f0f0f0", padx=10, pady=8)
        res.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        self.box = scrolledtext.ScrolledText(
            res, font=("Courier", 10),
            state="disabled", bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white")
        self.box.pack(fill="both", expand=True)

    # ── 쌍 추가 / 삭제 ────────────────────────────────────────────
    def _add_pair(self):
        row = PairRow(self._list_frame, self._remove_pair,
                      idx=len(self.pair_rows) + 1)
        self.pair_rows.append(row)

    def _remove_pair(self, row):
        if len(self.pair_rows) <= 1:
            return
        row.frame.destroy()
        self.pair_rows.remove(row)

    # ── 로그 ──────────────────────────────────────────────────────
    def _log(self, text):
        self.box.config(state="normal")
        self.box.insert("end", text + "\n")
        self.box.see("end")
        self.box.config(state="disabled")

    # ── 전체 시작 ─────────────────────────────────────────────────
    def _start(self):
        valid = [(r, r.orig_var.get(), r.susp_var.get())
                 for r in self.pair_rows
                 if r.orig_var.get() and r.susp_var.get()]
        if not valid:
            messagebox.showwarning("⚠", "원본과 비교본을 모두 선택한 쌍이 없습니다.")
            return

        self.run_btn.config(state="disabled")
        self.box.config(state="normal")
        self.box.delete("1.0", "end")
        self.box.config(state="disabled")
        for r, *_ in valid:
            r.reset()

        self._total = len(valid)
        self._done  = 0
        self._lock  = threading.Lock()
        self.status.set(f"0 / {self._total} 완료  —  {self._total}쌍 동시 분석 중...")

        for r, orig, susp in valid:
            threading.Thread(
                target=self._analyze_one,
                args=(r, orig, susp),
                daemon=True).start()

    # ── 쌍 1개 분석 (각 쌍이 독립 스레드에서 동시 실행) ─────────
    def _analyze_one(self, row, orig_path: str, susp_path: str):
        label    = Path(susp_path).stem
        thresh   = self.thresh.get()
        save_dir = os.path.dirname(susp_path) or "."

        def prog(v):
            self.after(0, lambda: row.set_prog(v))
        def st(msg, color="#2563eb"):
            self.after(0, lambda: row.set_status(msg, color))
        def log(text):
            self.after(0, lambda: self._log(text))

        try:
            st("🔄 오디오 추출")
            with tempfile.TemporaryDirectory() as tmp:
                wo = os.path.join(tmp, "orig.wav")
                ws = os.path.join(tmp, "susp.wav")
                if not extract_audio(orig_path, wo):
                    st("❌ ffmpeg", "#cc0000")
                    log(f"[{label}] ❌ ffmpeg 오류 — ffmpeg 설치 여부 확인")
                    return
                if not extract_audio(susp_path, ws):
                    st("❌ ffmpeg", "#cc0000")
                    log(f"[{label}] ❌ ffmpeg 오류 — ffmpeg 설치 여부 확인")
                    return

                st("🔄 특징 추출")
                fo, hop = extract_features(wo)
                fs, _   = extract_features(ws)

                st("🔄 탐색 중")
                segs = find_segments(fo, fs, hop, thresh, progress_cb=prog)

                if not segs:
                    st("✅ 없음", "#16a34a")
                    log(f"\n[{label}]  ✅  침해 의심 구간 없음")
                    save_result_txt([], orig_path, susp_path, thresh, save_dir)
                    return

                if self.do_video.get():
                    st("🔄 영상 검증")
                    for seg in segs:
                        seg["video_similarity"] = verify_video(
                            orig_path, susp_path, seg,
                            os.path.join(tmp, "frames"))
                else:
                    for seg in segs:
                        seg["video_similarity"] = -1.0

                # 결과 로그
                log(f"\n{'─' * 64}")
                log(f"  ⚠  [{label}]  —  {len(segs)}개 침해 의심 구간")
                log(f"{'─' * 64}")
                for i, s in enumerate(segs, 1):
                    dur = s["susp_end"] - s["susp_start"]
                    vs  = (f"  영상:{s['video_similarity']:.2f}"
                           if s["video_similarity"] >= 0 else "")
                    log(f"  [{i}] 비교본  "
                        f"{fmt_time(s['susp_start'])} ~ {fmt_time(s['susp_end'])}"
                        f"  ({dur:.1f}초)\n"
                        f"       원본    "
                        f"{fmt_time(s['orig_start'])} ~ {fmt_time(s['orig_end'])}\n"
                        f"       유사도: {s['similarity']:.2f}{vs}")

                p = save_result_txt(segs, orig_path, susp_path, thresh, save_dir)
                log(f"  📄 저장: {p}")
                st(f"⚠ {len(segs)}개", "#d97706")

        except Exception as e:
            import traceback
            st("❌ 오류", "#cc0000")
            log(f"[{label}] ❌ {e}\n{traceback.format_exc()}")

        finally:
            with self._lock:
                self._done += 1
                d, t = self._done, self._total
            self.after(0, lambda: self.status.set(f"{d} / {t} 완료"))
            if d >= t:
                self.after(0, lambda: (
                    self.run_btn.config(state="normal"),
                    self.status.set(f"✅  전체 {t}쌍 분석 완료")))


if __name__ == "__main__":
    App().mainloop()
