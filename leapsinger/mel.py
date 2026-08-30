"""同梱 NHVSing ボコーダと厳密一致する mel 計算(DiffSinger 44.1k/128/ln)。

各 DB 前処理パイプライン(ritsu/oniku/natsume/…/a speaker)の mel 計算を、ボコーダ学習時の
wav_to_mel と同一の recipe に統一するための共有関数。
= center=False + 事前 reflect pad (n_fft-hop)//2, hann, mel_basis @ |stft|, ln(clamp 1e-5)。
これで音響モデルの出力 mel が同梱ボコーダに in-distribution になる(resynth 実証済)。

DiffSinger/OpenUtau 標準値: 44100 / n_fft2048 / win2048 / hop512 / 128mel / fmin40 / fmax16000 / ln。
"""
import librosa
import numpy as np

# DiffSinger/NHVSing 標準(パイプラインからは定数で上書き可能だが既定はこれ)
SR      = 44_100
N_FFT   = 2048
WIN_LEN = 2048
HOP     = 512
N_MELS  = 128
F_MIN   = 40.0
F_MAX   = 16000.0

_MEL_BASIS_CACHE = {}


def _mel_basis(sr, n_fft, n_mels, fmin, fmax):
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key not in _MEL_BASIS_CACHE:
        _MEL_BASIS_CACHE[key] = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).astype(np.float64)
    return _MEL_BASIS_CACHE[key]


def wav_to_mel_nhv(wav, sr=SR, n_fft=N_FFT, hop=HOP, win=WIN_LEN,
                   n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX):
    """wav(mono, sr Hz)→ ln-mel [n_mels, T]。ボコーダの wav_to_mel と一致(center=False+reflect pad)。"""
    pad = (n_fft - hop) // 2
    yp = np.pad(wav.astype(np.float64), (pad, pad), mode='reflect')
    stft = librosa.stft(yp, n_fft=n_fft, hop_length=hop, win_length=win,
                        window='hann', center=False)
    mb = _mel_basis(sr, n_fft, n_mels, fmin, fmax)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        mel = mb @ np.abs(stft)
    return np.log(np.maximum(1e-5, mel)).astype(np.float32)   # [n_mels, T] 自然対数


_MEL_BASIS_TORCH_CACHE = {}


def wav_to_mag_nhv_torch(wav, n_fft=N_FFT, hop=HOP, win=WIN_LEN):
    """wav[B,L] or [L] torch → 線形振幅スペクトログラム |STFT| [B, n_fft//2+1, T](GPU可)。
    recipe は wav_to_mel_nhv_torch と同一(center=False + reflect pad (n_fft-hop)//2, hann, |STFT|)。
    数値安定のため fp32 で呼ぶこと(bf16 autocast 下で使わない)。"""
    import torch
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    pad = (n_fft - hop) // 2
    w = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
    window = torch.hann_window(win, periodic=True, device=wav.device, dtype=wav.dtype)
    S = torch.stft(w, n_fft, hop, win, window, center=False, return_complex=True)
    return S.abs()                                           # [B, freq, T] = numpy np.abs(stft)


def _mel_basis_torch(sr, n_fft, n_mels, fmin, fmax, device, dtype):
    import torch
    key = (sr, n_fft, n_mels, fmin, fmax, device, dtype)
    if key not in _MEL_BASIS_TORCH_CACHE:
        _MEL_BASIS_TORCH_CACHE[key] = torch.from_numpy(
            _mel_basis(sr, n_fft, n_mels, fmin, fmax)).to(device, dtype)
    return _MEL_BASIS_TORCH_CACHE[key]


def wav_to_mel_nhv_torch(wav, sr=SR, n_fft=N_FFT, hop=HOP, win=WIN_LEN,
                         n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX):
    """wav[B,L] or [L] torch → ln-mel [B,n_mels,T]。numpy wav_to_mel_nhv と厳密一致(GPU可)。
    target(教師mel)用途で勾配不要のため |stft| に eps を足さない(=numpy と一致)。STFT は
    数値安定のため fp32 で呼ぶこと(bf16 の autocast 下で使わない)。"""
    import torch
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    mag = wav_to_mag_nhv_torch(wav, n_fft, hop, win)        # [B, freq, T]
    mb  = _mel_basis_torch(sr, n_fft, n_mels, fmin, fmax, wav.device, wav.dtype)
    mel = torch.matmul(mb, mag)                             # [B, n_mels, T]
    return torch.log(torch.clamp(mel, min=1e-5))


def _cepstral_envelope(logmag, lifter_q):
    """周波数軸方向の実ケプストラム・リフタリングでスペクトル包絡(フォルマント)を推定。
    logmag [B, F, T] → env [B, F, T](低ケフレンシ lifter_q 本のみ残した平滑包絡)。
    ピッチのケフレンシ峰より低い lifter_q を選べば包絡=フォルマントのみを捉える。"""
    import torch
    F = logmag.shape[1]
    c = torch.fft.rfft(logmag, dim=1)                      # ケプストラム(周波数軸方向) [B, F//2+1, T] complex
    if lifter_q + 1 < c.shape[1]:
        c[:, lifter_q + 1:, :] = 0                         # 低ケフレンシのみ残す(リフタ)
    return torch.fft.irfft(c, n=F, dim=1)


def pitch_warp_mel_torch(wav, st, sr=SR, n_fft=N_FFT, hop=HOP, win=WIN_LEN,
                         n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
                         formant_preserve=True, lifter_q=30):
    """wav[B,L] を per-item 半音 st[B] だけ定長ピッチシフトした ln-mel [B,n_mels,T] を返す。
    位相不要(教師=振幅mel)なので |STFT| の周波数軸を r=2^(st/12) 倍ワープするだけ(GPU, no ISTFT)。
    formant_preserve=True: ケプストラム包絡補正で原フォルマントを復元(≈PSOLA・声質不変)。
      =False: 素ワープ(フォルマントも r 倍移動 = VTLP的)。フレーム数 T は不変(整合維持)。
    fp32・no_grad 前提。st=0 の item はワープ恒等 = wav_to_mel_nhv_torch と一致。"""
    import torch
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    mag = wav_to_mag_nhv_torch(wav, n_fft, hop, win)        # [B, F, T]
    B, Fbins, T = mag.shape
    if not torch.is_tensor(st):
        st = torch.full((B,), float(st), device=wav.device, dtype=wav.dtype)
    st = st.to(wav.device, wav.dtype).reshape(B)
    r = torch.pow(torch.tensor(2.0, device=wav.device, dtype=wav.dtype), st / 12.0)  # [B]

    # 周波数ワープ: out[b,f,t] = mag[b, f/r_b, t]。grid_sample で mag を [B,1,F,T] 画像扱い、
    # 高さ(F)を per-item に f/r_b で線形サンプル、幅(T)は恒等、Nyquist 超えは 0 埋め。
    f_idx = torch.arange(Fbins, device=wav.device, dtype=wav.dtype)        # [F]
    src   = f_idx[None, :] / r[:, None]                                    # [B, F] 参照元インデックス
    # align_corners=True 正規化: idx 0..F-1 → -1..+1
    gy = (src / (Fbins - 1)) * 2.0 - 1.0                                   # [B, F]
    gx = (torch.arange(T, device=wav.device, dtype=wav.dtype) / max(T - 1, 1)) * 2.0 - 1.0  # [T]
    grid = torch.empty(B, Fbins, T, 2, device=wav.device, dtype=wav.dtype)
    grid[..., 0] = gx[None, None, :]                                       # width = T (恒等)
    grid[..., 1] = gy[:, :, None]                                          # height = F (ワープ)
    warped = torch.nn.functional.grid_sample(
        mag.unsqueeze(1), grid, mode='bilinear',
        padding_mode='zeros', align_corners=True).squeeze(1)              # [B, F, T]

    if formant_preserve:
        log0 = torch.log(torch.clamp(mag,    min=1e-5))
        logw = torch.log(torch.clamp(warped, min=1e-5))
        env0 = _cepstral_envelope(log0, lifter_q)                         # 原フォルマント包絡
        envw = _cepstral_envelope(logw, lifter_q)                         # ワープ後包絡
        warped = torch.exp(logw + (env0 - envw))                          # 原包絡を復元(倍音櫛は r 倍)

    mb  = _mel_basis_torch(sr, n_fft, n_mels, fmin, fmax, wav.device, wav.dtype)
    mel = torch.matmul(mb, warped)
    return torch.log(torch.clamp(mel, min=1e-5))
