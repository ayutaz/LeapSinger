"""
phrase_cut.py

lab の静音区間(pau/br)を基準に音声をフレーズ単位で分割し、フレーズ単位の .npz を書き出す。

■ フレーズ分割アルゴリズム:
  静音区間 = pau/br のみ(cl/N/母音/子音 = 有音)。
  1. lab_segs を silence/voiced の極大 run に統合。
  2. 長さ ≥ split_silence_sec(既定0.5s)の静音 = CUT。CUT でフレーズグループを分ける。
  3. グループ内の content = 有音 + 内部の短い静音(<split)。内部 pau/br は句内に残す
     (=フレーズ途中に pau/br を含む長い句を生成できる能力を育てる)。
  4. content 長が max_dur_sec(既定10s)を超えたら、10s を越えた直後の ≥min_silence_sec の
     内部静音で追加分割(再帰)。≥min の静音が無ければ上限なしで長いまま(仕様)。
  5. 各 content の端に実際の静音を付与: lead/trail = min(隣接静音の実長, edge_max_sec=0.5)。
     例: 端静音0.3s→0.3s、5s→0.5s。直前静音が <lead_min_sec(0.1s)なら不足分を人工静音+pau で補う。
  6. 音量ゲート: 句の mel が p90(frame mean energy) < min_loudness(既定-8.0 ln-mel)なら保存しない
     (=ラベル↔音声ズレ由来の無音句を除去)。
  7. 境界を hop_size 倍数にスナップ、mel/f0/uv/phoneme をスライス、dur は frame から再構成。

■ 出力 npz フィールド(フレーズ単位):
    phoneme_ids : int16   [L_ph]     phoneme vocab ID
    dur_sec     : float32 [L_ph]     phoneme duration [s]
    f0_interp   : float32 [T]        interpolated F0 [Hz](ゼロなし)
    uv          : float32 [T]        voiced mask
    mel         : float32 [n_mel, T] log-mel spectrogram
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# ── 定数 ─────────────────────────────────────────────────────────────────────

_SILENCE_PHONES    = frozenset({'pau', 'sil', 'br', 'sp', 'sli'})   # 真の静音。cl/N/母音/子音 は有音
_MIN_SILENCE_SEC   = 0.1    # フレーズ境界候補・max_dur 追加分割点の最小静音長
_SPLIT_SILENCE_SEC = 0.5    # これ以上の静音で必ず区切る(CUT)。未満は句内に残す
_EDGE_MAX_SEC      = 0.5    # 句の端に含める静音の上限(実長を使い最大0.5s)
_LEAD_MIN_SEC      = 0.1    # 句頭に最低確保する静音(不足は人工静音で補う)
_MAX_DUR_SEC       = 10.0   # これを超えたら次の ≥min 静音で追加分割(ハード上限なし)
_MIN_LOUDNESS      = -8.0   # ln-mel の p90(frame mean) がこれ未満なら無音句として除去
_PAU_FADE_SEC      = 0.1    # pau 区間のフェード窓: 有音境界から 0.1s で gain 1→0(息継ぎ音除去)
_CL_PAU_SEC        = 0.3    # これより長い cl(促音)は誤ラベル休符とみなし静音扱い(短い cl は有音)
_MEL_FLOOR         = math.log(1e-5)   # 人工静音の mel 値(= ln(clamp) floor ≈ -11.51)


def _is_silence(ph: str, dur: float, silence_phones, cl_pau_sec: float) -> bool:
    """静音判定。pau/br/sil… は静音。cl は cl_pau_sec より長ければ静音(誤ラベル休符)扱い。"""
    if ph in silence_phones:
        return True
    if cl_pau_sec > 0 and ph == 'cl' and dur > cl_pau_sec:
        return True
    return False


# ── run 統合 ─────────────────────────────────────────────────────────────────

def _write_textgrid(path: Path, labels, durs, total_dur: float) -> None:
    """Praat TextGrid(1 IntervalTier 'phones')を書き出す。Praat で音声と重ねてタイミング確認用。"""
    N = len(labels)
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', '',
             'xmin = 0', f'xmax = {total_dur:.6f}', 'tiers? <exists>', 'size = 1', 'item []:',
             '    item [1]:', '        class = "IntervalTier"', '        name = "phones"',
             '        xmin = 0', f'        xmax = {total_dur:.6f}',
             f'        intervals: size = {N}']
    t = 0.0
    # labels と durs は同じ長さの想定。既存挙動を変えないため strict=False を明示する。
    for i, (lab, d) in enumerate(zip(labels, durs, strict=False), 1):
        xmin = t
        t += float(d)
        xmax = total_dur if i == N else t
        lines += [f'        intervals [{i}]:', f'            xmin = {xmin:.6f}',
                  f'            xmax = {xmax:.6f}', f'            text = "{lab}"']
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _merge_runs(lab_segs: list, silence_phones, cl_pau_sec: float = _CL_PAU_SEC) -> list:
    """連続する同種(静音/有音)を統合し (is_sil, start_sec, end_sec) の極大 run を返す。"""
    runs: list = []
    for s, e, ph in lab_segs:
        if e <= s:
            continue
        is_sil = _is_silence(ph, e - s, silence_phones, cl_pau_sec)
        if runs and runs[-1][0] == is_sil:
            runs[-1] = (is_sil, runs[-1][1], e)
        else:
            runs.append((is_sil, s, e))
    return runs


# ── フレーズ境界検出 ────────────────────────────────────────────────────────

def find_phrase_spans(
    lab_segs: list,
    total_sec: float,
    *,
    silence_phones = _SILENCE_PHONES,
    min_silence_sec:   float = _MIN_SILENCE_SEC,
    split_silence_sec: float = _SPLIT_SILENCE_SEC,
    edge_max_sec:      float = _EDGE_MAX_SEC,
    lead_min_sec:      float = _LEAD_MIN_SEC,
    max_dur_sec:       float = _MAX_DUR_SEC,
    cl_pau_sec:        float = _CL_PAU_SEC,
    hop_size:          int   = 512,
    sr:                int   = 44_100,
) -> list[tuple[int, int, int]]:
    """フレーズ区間 [(start_frame, end_frame, n_art_lead, n_art_trail), ...] を返す。
    境界は hop_size 倍数(mel フレーム境界)にスナップ済み。n_art_lead/n_art_trail>0 のとき
    その数だけ先頭に人工静音フレームを prepend する(cut_phrases 側で処理)。"""
    runs = _merge_runs(lab_segs, silence_phones, cl_pau_sec)
    if not runs:
        return []
    frame_rate = sr / hop_size
    total_frames = int(round(total_sec * frame_rate))

    # ── content(有音-有音, CUT静音で区切る)を抽出 ──
    contents: list = []   # (c_s, c_e)
    cur_s = None; cur_e = None
    for is_sil, s, e in runs:
        if is_sil and (e - s) >= split_silence_sec:      # CUT: 現 content を閉じる
            if cur_s is not None:
                contents.append((cur_s, cur_e)); cur_s = None
        elif not is_sil:                                  # 有音: content を伸ばす(末尾=最後の有音)
            if cur_s is None:
                cur_s = s
            cur_e = e
        # 短い静音(<split): content 内なら [cur_s,cur_e] に自然に含まれる。leading/trailing は無視
    if cur_s is not None:
        contents.append((cur_s, cur_e))
    if not contents:
        return []

    # ── max_dur 超えを 10s 直後の内部静音で追加分割(再帰) ──
    def _subsplit(c_s, c_e) -> list:
        if c_e - c_s <= max_dur_sec:
            return [(c_s, c_e)]
        cand = [(s, e) for is_sil, s, e in runs
                if is_sil and (e - s) >= min_silence_sec and s > c_s + max_dur_sec and e <= c_e]
        if not cand:
            return [(c_s, c_e)]                           # 切れる静音が無ければ長いまま(上限なし)
        s0, e0 = cand[0]                                  # 10s を越えた最初の内部静音
        return _subsplit(c_s, s0) + _subsplit(e0, c_e)

    split_contents: list = []
    for c_s, c_e in contents:
        split_contents.extend(_subsplit(c_s, c_e))

    # ── 端に実静音を付与(lead/trail = min(隣接静音, edge_max))。不足分は人工静音 ──
    def _sil_before(t):     # end == t の静音 run の長さ
        for is_sil, s, e in runs:
            if is_sil and abs(e - t) < 1e-6:
                return e - s
        return 0.0
    def _sil_after(t):      # start == t の静音 run の長さ
        for is_sil, s, e in runs:
            if is_sil and abs(s - t) < 1e-6:
                return e - s
        return 0.0

    spans: list[tuple[int, int, int, int]] = []
    for c_s, c_e in split_contents:
        prev_sil = _sil_before(c_s)
        next_sil = _sil_after(c_e)
        lead  = min(prev_sil, edge_max_sec)
        trail = min(next_sil, edge_max_sec)
        # 実静音が lead_min 未満なら人工静音で補う(先頭・末尾とも。フレーズは必ず pau で始まり
        # pau で終わる)
        art_lead  = max(0.0, lead_min_sec - prev_sil)
        art_trail = max(0.0, lead_min_sec - next_sil)
        span_s = max(0.0, c_s - lead)
        span_e = min(total_sec, c_e + trail)
        # フレームスナップ: 端が静音内にある側は静音の内側へ丸める(隣接フレーズの有音の
        # 欠片を端フレームに掴み無音化を壊さないため)。静音が無い側は content を覆う方向。
        if lead > 0:
            s_fr = -(-int(span_s * sr) // hop_size)        # 切り上げ(静音内側)
        else:
            s_fr = int(span_s * sr) // hop_size            # 切り捨て(content を覆う)
        if trail > 0:
            e_fr = int(span_e * sr) // hop_size            # 切り捨て(静音内側)
        else:
            e_fr = -(-int(span_e * sr) // hop_size)        # 切り上げ(content を覆う)
        e_fr = min(e_fr, total_frames)
        n_art_lead  = int(round(art_lead * frame_rate))
        n_art_trail = int(round(art_trail * frame_rate))
        if e_fr > s_fr:
            spans.append((s_fr, e_fr, n_art_lead, n_art_trail))
    return spans


def build_sil_mask(
    lab_segs: list,
    total_frames: int,
    frame_rate: float,
    *,
    silence_phones = _SILENCE_PHONES,
    cl_pau_sec: float = _CL_PAU_SEC,
) -> np.ndarray:
    """ラベル基準の静音(pau/長cl)フレームマスク bool[T]。"""
    sil = np.zeros(total_frames, dtype=bool)
    for s, e, ph in lab_segs:
        if _is_silence(ph, e - s, silence_phones, cl_pau_sec):
            a = max(0, int(round(s * frame_rate)))
            b = min(total_frames, int(round(e * frame_rate)))
            if b > a:
                sil[a:b] = True
    return sil


def clip_pau_gain(sil_clip: np.ndarray, fade_frames: int) -> np.ndarray:
    """1フレーズ(クリップ)内の pau 無音化ゲイン g[n]。距離はクリップ内の最近傍有音まで
    → フレーズ端の pau は片側傾斜(クリップ端=必ず完全無音)、内部 pau のみ両側傾斜。
    端の sil run が fade_frames より短い場合は傾斜を run 長に圧縮し端点で 0 に到達させる。"""
    n = len(sil_clip)
    if not sil_clip.any():
        return np.ones(n, dtype=np.float32)
    if fade_frames <= 0 or sil_clip.all():
        return (~sil_clip).astype(np.float32)
    vi = np.flatnonzero(~sil_clip)
    # 最近傍の有音フレームまでの距離 d を O(n) で（[n, n_voiced] 距離行列は作らない）。
    # 左方向=maximum.accumulate / 右方向=反転 minimum.accumulate で最近傍有音 index を得て距離化。
    idx    = np.arange(n)
    voiced = ~sil_clip
    last_v = np.maximum.accumulate(np.where(voiced, idx, -1))          # 左側の最近傍有音 index(-1=無)
    nxt    = np.where(voiced, idx, n)
    next_v = np.minimum.accumulate(nxt[::-1])[::-1]                    # 右側の最近傍有音 index(n=無)
    d = np.minimum(np.where(last_v >= 0, idx - last_v, n),
                   np.where(next_v < n,  next_v - idx, n))             # 端は自動で片側距離
    g  = np.clip(1.0 - d / fade_frames, 0.0, 1.0).astype(np.float32)
    # 先頭の sil run: 傾斜長 = min(fade, run長) で端点=0 を保証(片側傾斜)
    lead = int(vi[0])                      # 先頭 sil run 長
    if lead > 0:
        L = min(fade_frames, lead)
        dd = lead - np.arange(lead)        # 有音境界までの距離(端で最大)
        g[:lead] = np.minimum(g[:lead], np.clip(1.0 - dd / L, 0.0, 1.0))
    trail = n - 1 - int(vi[-1])            # 末尾 sil run 長
    if trail > 0:
        L = min(fade_frames, trail)
        dd = np.arange(1, trail + 1)       # 有音境界からの距離(端で最大=trail)
        g[n - trail:] = np.minimum(g[n - trail:], np.clip(1.0 - dd / L, 0.0, 1.0))
    return g


# ── フレーズ npz 生成 ─────────────────────────────────────────────────────────

def cut_phrases(
    name:      str,
    lab_segs:  list,
    ph_ids:    np.ndarray,   # int16  [L_ph]
    dur_sec:   np.ndarray,   # float32[L_ph]
    f0_raw:    np.ndarray | None,   # float32[T]  0=unvoiced(曲全体、補完前)。span_data 使用時 None 可
    uv:        np.ndarray | None,   # float32[T]                          。span_data 使用時 None 可
    mel:       np.ndarray | None,   # float32[n_mel, T]                   。span_data 使用時 None 可
    out_dir:   Path,
    *,
    span_data = None,               # callable(s_fr, e_fr) -> (mel[n_mel,n], f0[n], uv[n])
                                    # フレーズ単位に wav をカット→pau無音化→mel/F0 抽出する側が渡す。
                                    # 指定時は mel/f0_raw/uv の全体配列は不要(total_frames 必須)。
    total_frames: int | None = None,  # span_data 使用時の曲全体フレーム数
    silence_phones = _SILENCE_PHONES,
    min_silence_sec:   float = _MIN_SILENCE_SEC,
    split_silence_sec: float = _SPLIT_SILENCE_SEC,
    edge_max_sec:      float = _EDGE_MAX_SEC,
    lead_min_sec:      float = _LEAD_MIN_SEC,
    max_dur_sec:       float = _MAX_DUR_SEC,
    min_loudness:      float = _MIN_LOUDNESS,
    pau_fade_sec:      float = _PAU_FADE_SEC,   # 0 で無効
    cl_pau_sec:        float = _CL_PAU_SEC,     # これより長い cl は静音扱い(cut+fade)
    hop_size:          int   = 512,
    sr:                int   = 44_100,
    pau_id:            int   = 0,     # 人工静音に使う pau の vocab id
    spans:             list | None = None,   # 事前計算済み spans(F0をフレーズ単位抽出する側と共有)
    verbose:           bool  = False,
) -> int:
    """フレーズ単位で npz を書き出す。Returns: 書き出したフレーズ数。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    T_frames   = mel.shape[1] if mel is not None else int(total_frames)
    frame_rate = sr / hop_size
    total_sec  = T_frames / frame_rate

    if spans is None:
        spans = find_phrase_spans(
            lab_segs, total_sec,
            silence_phones=silence_phones, min_silence_sec=min_silence_sec,
            split_silence_sec=split_silence_sec, edge_max_sec=edge_max_sec,
            lead_min_sec=lead_min_sec, max_dur_sec=max_dur_sec, cl_pau_sec=cl_pau_sec,
            hop_size=hop_size, sr=sr,
        )
    if not spans:
        spans = [(0, T_frames, 0, 0)]

    # 音素の cumulative フレーム境界(score→frame)
    cum_sec      = np.concatenate([[0.0], np.cumsum(dur_sec)])
    ph_frame_bnd = np.round(cum_sec * frame_rate).astype(int)
    ph_frame_bnd[-1] = T_frames

    # 曲全体の静音(pau/br/sil…)フレームマスク(lab 基準)。pau フェード窓に使う
    sil_mask = np.zeros(T_frames, dtype=bool)
    for s, e, ph in lab_segs:
        if _is_silence(ph, e - s, silence_phones, cl_pau_sec):
            a = max(0, int(round(s * frame_rate)))
            b = min(T_frames, int(round(e * frame_rate)))
            if b > a:
                sil_mask[a:b] = True
    fade_frames = max(1, int(round(pau_fade_sec * frame_rate))) if pau_fade_sec > 0 else 0

    n_written = 0
    n_silent  = 0
    idx = 0
    for s_fr, e_fr, n_art, n_art_trail in spans:
        if e_fr <= s_fr:
            continue
        n_frames = e_fr - s_fr

        extra = {}
        wav_int16 = None
        if span_data is not None:
            # フレーズ単位パス: wav カット→pau無音化(端=片側)→mel/F0 抽出済みを受け取る。
            # mel は wav 側で無音化済みなので mel フェードは行わない。
            # 4番目(任意)は npz に追記するメタ(F0 レンジ等)。5番目(任意)は sv_raw の生 clip wav。
            _out = span_data(s_fr, e_fr)
            mel_slice, f0_phrase_raw, uv_slice = _out[:3]
            if len(_out) > 3 and _out[3]:
                extra = {k: np.float32(v if v is not None else np.nan)
                         for k, v in _out[3].items()}
            if len(_out) > 4 and _out[4] is not None:        # sv_raw: 生 clip wav を int16 で句npzに保存
                wav_int16 = (np.clip(_out[4], -1.0, 1.0) * 32767.0).astype(np.int16)
            # ── 音量ゲート: 無音句(ラベル↔音声ズレ)を除去 ──
            if float(np.percentile(mel_slice.mean(axis=0), 90)) < min_loudness:
                n_silent += 1
                continue
        else:
            mel_slice = mel[:, s_fr:e_fr]

            # ── 音量ゲート: 無音句(ラベル↔音声ズレ)を除去 ──
            if float(np.percentile(mel_slice.mean(axis=0), 90)) < min_loudness:
                n_silent += 1
                continue

            uv_slice = uv[s_fr:e_fr]

            # ── pau フェード窓: 有音境界から fade_frames で gain 1→0(息継ぎ音を除去) ──
            # g[frame] = max(0, 1 - d/fade)、d = 最も近い有音(非pau)フレームまでの距離。
            # 端pau=片側フェード / 内部pau=両側フェード / 中間=完全無音(floor)を統一的に実現。
            if fade_frames > 0:
                sil_phrase = sil_mask[s_fr:e_fr]
                if sil_phrase.any() and (~sil_phrase).any():
                    vi = np.flatnonzero(~sil_phrase)                       # 有音フレーム index
                    d  = np.abs(np.arange(n_frames)[:, None] - vi[None, :]).min(axis=1)
                    g  = np.clip(1.0 - d / fade_frames, 0.0, 1.0)
                    logg = np.log(np.clip(g, 1e-12, 1.0)).astype(mel_slice.dtype)
                    mel_slice = np.maximum(_MEL_FLOOR, mel_slice + logg[None, :])

            # フレーズ単位 f0 補間(両端pauは端の voiced 値で一定、内部短pauは線形)
            f0_phrase_raw = f0_raw[s_fr:e_fr]

        voiced_phrase = uv_slice > 0.5
        if voiced_phrase.any():
            x = np.arange(n_frames, dtype=float)
            f0_slice = np.interp(x, x[voiced_phrase], f0_phrase_raw[voiced_phrase]).astype(np.float32)
        else:
            f0_slice = f0_phrase_raw.copy()

        # ── このフレーズに重なる音素を抽出 ──
        ph_mask = (ph_frame_bnd[1:] > s_fr) & (ph_frame_bnd[:-1] < e_fr)
        if not ph_mask.any():
            continue
        sel_ids   = ph_ids[ph_mask]
        sel_starts = np.maximum(ph_frame_bnd[:-1][ph_mask], s_fr)
        sel_ends   = np.minimum(ph_frame_bnd[1:][ph_mask],  e_fr)
        sel_dur_fr = np.maximum(sel_ends - sel_starts, 1)
        sel_dur_s  = (sel_dur_fr / frame_rate).astype(np.float32)
        # 末尾補正(frame スナップ由来の微差を吸収)
        diff = n_frames / frame_rate - float(sel_dur_s.sum())
        sel_dur_s[-1] = max(1.0 / frame_rate, sel_dur_s[-1] + diff)

        def _pad_extra(cur_len: int, n_pre: int, n_post: int):
            """extra 内のフレーム長配列(f0_alg_* 等)も mel/f0 と同じだけパディングして整列を保つ。
            2-D [C, cur_len](ピッチ水増しの mel_up05 等)は mel_slice と同じ人工静音で拡張。"""
            for k, v in extra.items():
                if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == cur_len:
                    extra[k] = np.concatenate([np.zeros(n_pre, dtype=v.dtype), v,
                                               np.zeros(n_post, dtype=v.dtype)])
                elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == cur_len:
                    fill = _MEL_FLOOR if k.startswith('mel') else 0.0
                    extra[k] = np.concatenate(
                        [np.full((v.shape[0], n_pre),  fill, dtype=v.dtype), v,
                         np.full((v.shape[0], n_post), fill, dtype=v.dtype)], axis=1)

        # ── 人工静音を先頭に prepend(実静音 < lead_min の場合。フレーズは必ず pau で始まる) ──
        if n_art > 0:
            _pad_extra(len(f0_slice), n_art, 0)
            mel_slice = np.concatenate(
                [np.full((mel_slice.shape[0], n_art), _MEL_FLOOR, dtype=mel_slice.dtype), mel_slice], axis=1)
            f0_slice  = np.concatenate([np.zeros(n_art, dtype=np.float32), f0_slice])
            uv_slice  = np.concatenate([np.zeros(n_art, dtype=np.float32), uv_slice])
            sel_ids   = np.concatenate([[pau_id], sel_ids])
            sel_dur_s = np.concatenate([[np.float32(n_art / frame_rate)], sel_dur_s])

        # ── 人工静音を末尾に append(実静音 < lead_min の場合。フレーズは必ず pau で終わる) ──
        if n_art_trail > 0:
            _pad_extra(len(f0_slice), 0, n_art_trail)
            mel_slice = np.concatenate(
                [mel_slice, np.full((mel_slice.shape[0], n_art_trail), _MEL_FLOOR, dtype=mel_slice.dtype)], axis=1)
            f0_slice  = np.concatenate([f0_slice, np.zeros(n_art_trail, dtype=np.float32)])
            uv_slice  = np.concatenate([uv_slice, np.zeros(n_art_trail, dtype=np.float32)])
            sel_ids   = np.concatenate([sel_ids, [pau_id]])
            sel_dur_s = np.concatenate([sel_dur_s, [np.float32(n_art_trail / frame_rate)]])

        np.savez(
            str(out_dir / f'{name}_{idx:04d}.npz'),
            phoneme_ids = sel_ids.astype(np.int16),
            dur_sec     = sel_dur_s.astype(np.float32),
            f0_interp   = f0_slice,
            uv          = uv_slice,
            mel         = mel_slice,
            # NOTE: speaker/style identity is intentionally NOT written here. The shard is
            # identity-free; spk_id/style_id are assigned per-DB at TRAIN time (dataset spk_map/
            # style_map), so the mapping can change without re-preprocessing.
            **({'wav': wav_int16} if wav_int16 is not None else {}),   # sv_raw: 句wav(int16)
            **extra,
        )
        n_written += 1
        idx += 1

    if verbose and n_silent:
        print(f'    [{name}] 無音ゲート除去 {n_silent} 句 / 書出 {n_written} 句')
    return n_written
