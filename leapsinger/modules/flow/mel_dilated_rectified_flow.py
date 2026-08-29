"""
MelDilatedRectifiedFlow: mel 空間 rectified flow（backbone = DiffWave型 dilated conv）。

mel を [B, mel_bins, T] のまま扱う。学習=compute_loss（速度場回帰）、推論=inference（少数ステップ
Euler 積分）。条件 cond は1系統（CFG なし）。
"""
import torch
import torch.nn as nn

from leapsinger.modules.backbones.dilated_conv import DilatedConvDenoiser


class MelDilatedRectifiedFlow(nn.Module):
    def __init__(
        self,
        vmin: float = -6.0,
        vmax: float = 0.0,
        mel_bins: int = 80,
        dim_cond: int = 256,
        ch: int = 256,
        n_cycles: int = 3,
        kernel: int = 3,
        loss_type: str = 'l1',   # 'l1'(従来・既定) | 'l2'(openvpi と同じ速度場 MSE)
        dilations=None,          # 明示 dilation 列（指定時は n_cycles を無視。backbone A/B 用）
    ):
        super().__init__()
        assert loss_type in ('l1', 'l2'), loss_type
        self.vmin = vmin
        self.vmax = vmax
        self.mel_bins = mel_bins
        self.loss_type = loss_type
        self.velocity_fn = DilatedConvDenoiser(
            mel_bins=mel_bins, ch=ch, dim_cond=dim_cond, n_cycles=n_cycles, kernel=kernel,
            dilations=dilations,
        )

    def _norm(self, x):                            # [B, mel_bins, T] → [-1,1]
        return (x - self.vmin) / (self.vmax - self.vmin) * 2 - 1

    def _denorm(self, x):
        return (x + 1) / 2 * (self.vmax - self.vmin) + self.vmin

    # ── training ──────────────────────────────────────────────────────────────
    def compute_loss(self, target, cond, x0_init=None, mask=None):
        """target [B, mel_bins, T], cond [B, dim_cond, T]
        → (scalar loss, x1_pred [B, mel_bins, T])。
        loss は速度場に対する L1（既定）または L2（loss_type='l2'）。
        x1_pred は rectified flow の 1-step clean 予測（x_t + (1-t)·v_pred を逆正規化）。
        x0_init: flow の出発点(正規化域)。None のときは Gaussian prior にフォールバック。
        mask: [B, T] bool（True=有効フレーム）。None=全フレーム平均。指定時のみパディング枠を除外して平均。"""
        x1 = self._norm(target)
        x0 = torch.randn_like(x1) if x0_init is None else x0_init
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t = x0 + t[:, None, None] * (x1 - x0)
        v_pred = self.velocity_fn(x_t, t, cond)
        v_gt = x1 - x0
        diff = v_pred - v_gt
        if mask is None:
            loss = diff.abs().mean() if self.loss_type == 'l1' else diff.pow(2).mean()
        else:
            e = diff.abs() if self.loss_type == 'l1' else diff.pow(2)
            m = mask[:, None, :].to(e.dtype)                 # [B,1,T]
            loss = (e * m).sum() / m.sum().clamp(min=1.0) / e.shape[1]
        x1_pred = self._denorm(x_t + (1.0 - t[:, None, None]) * v_pred)
        return loss, x1_pred

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def inference(self, cond, num_steps: int = 10, algorithm: str = 'euler', x0_init=None):
        """cond [B, dim_cond, T] → mel [B, mel_bins, T]。
        x0_init: flow の出発点(正規化域)。None のときは Gaussian prior にフォールバック。"""
        B, _, T = cond.shape
        x = torch.randn(B, self.mel_bins, T, device=cond.device) if x0_init is None else x0_init
        dt = 1.0 / num_steps

        def v(xx, tt):
            return self.velocity_fn(xx, tt, cond)

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=cond.device)
            if algorithm == 'euler':
                x = x + v(x, t) * dt
            elif algorithm == 'rk2':
                k1 = v(x, t)
                k2 = v(x + 0.5 * k1 * dt, (t + 0.5 * dt).clamp(0, 1))
                x = x + k2 * dt
            elif algorithm == 'rk4':
                k1 = v(x, t)
                k2 = v(x + 0.5 * k1 * dt, (t + 0.5 * dt).clamp(0, 1))
                k3 = v(x + 0.5 * k2 * dt, (t + 0.5 * dt).clamp(0, 1))
                k4 = v(x + k3 * dt, (t + dt).clamp(0, 1))
                x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            else:
                raise ValueError(f'Unknown algorithm: {algorithm}')
        return self._denorm(x)
