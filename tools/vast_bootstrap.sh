#!/usr/bin/env bash
# vast.ai (Linux) の新規インスタンスに LeapSinger の学習環境を作る。
#
#   # インスタンス上で。リポジトリを clone するところから:
#   curl -LsSf https://raw.githubusercontent.com/ayutaz/LeapSinger/feature/svc/tools/vast_bootstrap.sh | bash
#
#   # すでにコードを置いてある場合はリポジトリ直下で:
#   bash tools/vast_bootstrap.sh
#
# 環境変数:
#   REPO    clone 元 (既定: https://github.com/ayutaz/LeapSinger.git)
#   BRANCH  clone するブランチ (既定: feature/svc)
#   DIR     clone 先 (既定: $HOME/LeapSinger)
#   EXTRAS  uv sync に渡す extra (既定: "--extra train")
#   RMVPE   1 なら RMVPE の重み(約181MB)も落とす。前処理を回すなら必要
set -euo pipefail

REPO="${REPO:-https://github.com/ayutaz/LeapSinger.git}"
BRANCH="${BRANCH:-feature/svc}"
DIR="${DIR:-$HOME/LeapSinger}"
EXTRAS="${EXTRAS:---extra train}"
RMVPE="${RMVPE:-0}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1. マシン"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "nvidia-smi が無い/失敗。GPU インスタンスか確認すること" >&2; exit 1; }
df -h "$HOME" | tail -1
echo "note: torch cu130 + nvidia wheel だけで 8GB 前後使う。ディスクは 40GB 以上を推奨"

say "2. uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

say "3. リポジトリ"
if [ -f "pyproject.toml" ] && [ -d "leapsinger" ]; then
  echo "カレントがリポジトリなのでそのまま使う: $(pwd)"
else
  if [ ! -d "$DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO" "$DIR"
  else
    git -C "$DIR" fetch origin "$BRANCH" && git -C "$DIR" checkout "$BRANCH" && git -C "$DIR" pull
  fi
  cd "$DIR"
fi
git log --oneline -1

say "4. 依存 (uv.lock どおり。Python 3.13 は uv が用意する)"
# shellcheck disable=SC2086
uv sync $EXTRAS
uv run python -VV

say "5. GPU 疎通"
uv run python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda build     :", torch.version.cuda)
print("is_available   :", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA が見えていない。ドライバかイメージを確認すること"
print("device         :", torch.cuda.get_device_name(0))
print("capability     :", torch.cuda.get_device_capability(0))
print("bf16           :", torch.cuda.is_bf16_supported())
print("VRAM           : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
try:
    import triton
    print("triton         :", triton.__version__)
except ImportError:
    print("triton         : なし（torch.compile は使えない）")
PY

say "6. 倍音和の torch.compile 経路が効いているか"
# Windows は Triton wheel が無くループ版に落ちる。Linux では compile が効くはずで、
# ここが False のままなら励起が 3〜4 倍遅い状態のまま学習することになる。
uv run python - <<'PY'
import time, torch
import leapsinger.modules.harmonic_excitation as H
f0 = torch.full((2, 400), 8.0, device="cuda")
uv = torch.ones(2, 400, device="cuda")
H.harmonic_wave(f0, uv, n_harm=256, sr=44100, hop=256)   # 初回で compile される
torch.cuda.synchronize()
t = time.time()
for _ in range(10):
    H.harmonic_wave(f0, uv, n_harm=256, sr=44100, hop=256)
torch.cuda.synchronize()
ms = (time.time() - t) / 10 * 1e3
active = H._harm_sum_c is not None
print(f"compiled path active: {active}   harmonic_wave: {ms:.1f} ms/call")
if not active:
    print("!! ループ版にフォールバックしている。上のログの理由を確認すること")
    print("   (意図的に切るなら LEAPSINGER_EXC_COMPILE=0)")
PY

say "7. 単体テスト"
uv run python -m unittest test_svc_model

if [ "$RMVPE" = "1" ]; then
  say "8. RMVPE の重み (前処理用・約181MB)"
  uv run python -c "from preprocess.algorithms.rmvpe import get_model_path; print(get_model_path(None))"
fi

say "完了"
cat <<'EOS'
次にやること:
  - データ(shard)をインスタンスへ置く。data/<db>/ 配下。
  - 学習:
      uv run python -m train --config configs/svc_base.yaml \
        --data_dirs data/<db> --run_name <run> --out_root log --device cuda
  - 同じコマンドの再実行で log/<run>/ckpt_*.pt の最新から自動再開する。
    別実験では必ず --run_name を変えること。
  - log/ は消えるので、ckpt と events は定期的に外へ退避すること。
EOS
