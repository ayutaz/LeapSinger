#!/usr/bin/env python3
"""blind preference test を用意し、集計する（[実行計画](../doc/svc-plan.md) M5 ゴール 3）。

    # 1. 聴く用のファイルと採点シートを作る（どちらが A かは分からない形で並ぶ）
    uv run python tools/blind_test.py prepare --a out/m5/leapsvc --b out/m5/seedvc \
      --a-name leapsvc --b-name seedvc --out out/m5/blind --seed 0

    # 2. out/m5/blind/sheet.csv の vote 列に A / B / tie を書いてから
    uv run python tools/blind_test.py tally --sheet out/m5/blind/sheet.csv \
      --key out/m5/blind/key.json --out out/m5/blind/result.json

**評価者は 1 名（開発者）です**（2026-09-01 決定）。**N=1 の非公式 preference test** として
報告し、**MOS とは呼びません**。それでも blind の条件は満たします。

| 条件 | 実装 |
|---|---|
| ラベルを隠す | 提示ファイル名は `pair03_A.wav` の形。**system 名が出ない** |
| 左右を入れ替える | clip ごとに A / B の割り当てを randomize。常に片方が A だと順序の癖が preference に化ける |
| 順番を混ぜる | clip の並びも shuffle。曲順で聴くと後半に慣れが出る |
| 復元できる | 割り当ては `key.json` に残す。**採点シートには入れない** |

**引き分けと未記入を捨てません。** 捨てると「差が無かった」ことが見えなくなります。

**符号検定の p 値は参考値**です。評価者 1 名なので、独立なのは clip であって人ではありません。
**「preferred」と書くときは必ず N=1 と clip 数を併記します。**
"""
from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import Any


def match_loudness(a, b, *, peak_limit: float = 0.95):
    """2 本の波形を**同じ RMS へ揃える**。

    **揃えないと blind になりません。** 実測で LeapSVC の出力は Seed-VC の 5.4 倍大きく
    （RMS 0.143 対 0.027）、**音量だけで system を当てられる状態**でした。人は大きいほうを
    好むので、preference が音量の選好に化けます。

    **クリップさせません。** 揃えるために持ち上げて 1.0 を超えると、歪みが preference に
    化けます。両方が `peak_limit` に収まる範囲で、できるだけ大きい共通の RMS を選びます。
    """
    import numpy as np

    xs = [np.asarray(x, dtype=np.float32).copy() for x in (a, b)]
    rms = [float(np.sqrt(np.mean(x ** 2))) for x in xs]
    peak = [float(np.abs(x).max()) for x in xs]
    if min(rms) <= 1e-9:
        return tuple(xs)                       # 無音が混ざっていたら触らない

    # 各波形が peak_limit を超えない最大の RMS。その最小値を共通の目標にする。
    headroom = [peak_limit / p * r for p, r in zip(peak, rms, strict=True) if p > 1e-9]
    target = min(headroom) if headroom else min(rms)
    return tuple((x * (target / r)).astype(np.float32) for x, r in zip(xs, rms, strict=True))


def concat_pair(a, b, *, sr: int, gap_sec: float = 0.7):
    """A -> 無音 -> B を 1 本に繋ぐ。**音量は触りません**（揃えた意味が消えるため）。

    26 ペアを A/B 別ファイルで聴くと切り替えの手間が大きいので、繋いだ形も置きます。
    """
    import numpy as np

    gap = np.zeros(int(gap_sec * sr), dtype=np.float32)
    return np.concatenate([np.asarray(a, dtype=np.float32),
                           gap,
                           np.asarray(b, dtype=np.float32)]).astype(np.float32)


SHEET_HEADER = "pair,clip,vote  # vote に A / B / tie を書く"
_VALID_VOTES = ("", "A", "B", "tie")


def find_source(root, clip: str):
    """`<曲>__<clip>_source.wav` を 1 つだけ取る。

    **前方一致にしません。** `unseen1` が `unseen10` に当たると、**別 clip の変換元**を
    聴かせることになります。区切りの `__` と `_source.wav` を含めて厳密に照合します。

    変換元は**両方の系で同じ入力**なので、聴かせても A / B の正体は漏れません。
    """
    from pathlib import Path as _P

    hits = sorted(_P(root).glob(f"*__{clip}_source.wav"))
    if len(hits) > 1:
        raise ValueError(f"{clip}: 変換元が {len(hits)} 個あります（{[h.name for h in hits]}）")
    return hits[0] if hits else None


_ALLOWED_DIRS = ("audio/", "paired/", "context/")


def _check_path(path: str, what: str) -> str:
    """ページから指してよい場所か検査する。

    **blind の穴を 2 つ塞ぎます。**

    1. `out/m5/leapsvc/...` のような元ディレクトリを直接指すと、**パスに system 名が
       出ます**。中立な名前で `context/` へ複写したものだけを指させます。
    2. 上限（`*_vocoder_only.wav`）は **LeapSVC 自身のボコーダーの音**です。参照として
       聴かせると、そのボコーダーの癖で A / B のどちらが LeapSVC かを当てられます。
    """
    p = str(path).replace("\\", "/")
    if "vocoder_only" in p:
        raise ValueError(f"{what}: 上限（{p}）は聴かせられません。"
                         "LeapSVC 自身のボコーダーの音なので、系を当てる手がかりになります")
    if not p.startswith(_ALLOWED_DIRS):
        raise ValueError(f"{what}: {p} は blind ディレクトリの外です。"
                         f"パスに system 名が出るので、{'/'.join(_ALLOWED_DIRS)} へ"
                         "中立な名前で複写してから指すこと")
    return p


def listen_page(rows: Sequence[dict], *, title: str = "blind preference test",
                references: Sequence[dict] = ()) -> str:
    """ブラウザで聴いて投票するページを組み立てる（`out/m5/blind/listen.html`）。

    **key.json の行を渡すと落とします。** key には `A` / `B` に system 名が入っており、
    ページへ埋めると **blind が崩れます**。採点シート側の行（`pair` / `clip` / `vote`）
    だけを受け取ります。

    音声は `<audio src="audio/pairNN_A.wav">` の相対参照です。`file://` では `fetch`
    が塞がれるので、行は **HTML に直接埋め込みます**（ページ単体で開けること）。

    投票は localStorage に持ち、`sheet.csv` と同じ形で書き出します。**並べ替えません** ―
    prepare が shuffle 済みで、ページが並べ直すと曲順の癖が preference に戻ります。
    """
    import json

    rows = list(rows)
    if not rows:
        raise ValueError("行がありません")
    clean = []
    for r in rows:
        if "A" in r or "B" in r:
            raise ValueError(f"key.json の行を渡しています（{r.get('pair')}）。"
                             "答えがページに埋まるので受け取れません")
        vote = str(r.get("vote") or "").strip()
        if vote not in _VALID_VOTES:
            raise ValueError(f"{r.get('pair')}: vote が {vote!r}。A / B / tie / 空 のみ")
        pair = str(r["pair"])
        # **音声の場所は data に持たせます。** JS で組み立てるとページ本文にファイル名が
        # 出ず、欠けているものを目視で確認できません。
        item = {"pair": pair, "clip": str(r["clip"]), "vote": vote,
                "a": f"audio/{pair}_A.wav", "b": f"audio/{pair}_B.wav",
                "ab": f"paired/{pair}_AB.wav"}
        if r.get("source"):
            item["source"] = _check_path(r["source"], f"{pair} の変換元")
        clean.append(item)

    refs = [{"label": str(x["label"]), "path": _check_path(x["path"], "参照")}
            for x in references]

    data = json.dumps(clean, ensure_ascii=False, indent=1)
    return _PAGE_TEMPLATE.replace("__TITLE__", title) \
                         .replace("__HEADER__", json.dumps(SHEET_HEADER, ensure_ascii=False)) \
                         .replace("__REFS__", json.dumps(refs, ensure_ascii=False, indent=1)) \
                         .replace("__ROWS__", data)


_PAGE_TEMPLATE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
 :root{color-scheme:dark;--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--line:#2a2e35;
       --acc:#7cc4ff;--ok:#7ddc8f;--tie:#d9b45b}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.65 system-ui,"Segoe UI","Yu Gothic UI",sans-serif}
 .wrap{max-width:760px;margin:0 auto;padding:24px 20px 64px}
 h1{font-size:19px;margin:0 0 4px}
 .sub{color:var(--dim);font-size:13px;margin:0 0 20px}
 .bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:14px 0 6px}
 .bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .2s}
 .count{color:var(--dim);font-size:13px;display:flex;justify-content:space-between}
 .card{border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0;
       background:#191c21}
 .pair{font-size:22px;font-weight:600;letter-spacing:.02em}
 .clip{color:var(--dim);font-size:13px;margin-top:2px}
 .row{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
 button{font:inherit;color:var(--fg);background:#232830;border:1px solid var(--line);
        border-radius:9px;padding:11px 16px;cursor:pointer}
 button:hover{border-color:var(--acc)}
 button:disabled{opacity:.4;cursor:default}
 .play{min-width:120px}
 .play.on{border-color:var(--acc);background:#1d2c3a}
 .vote{flex:1;min-width:110px;font-weight:600}
 .vote[data-on="1"]{background:#1e3326;border-color:var(--ok);color:var(--ok)}
 .vote.tie[data-on="1"]{background:#332c18;border-color:var(--tie);color:var(--tie)}
 .nav{display:flex;justify-content:space-between;gap:10px;margin-top:8px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:6px;
       margin-top:10px}
 .grid button{padding:7px 4px;font-size:12px;text-align:center}
 .grid button.done{border-color:var(--ok);color:var(--ok)}
 .grid button.cur{background:#1d2c3a;border-color:var(--acc)}
 .keys{color:var(--dim);font-size:12px;margin-top:14px}
 .warn{border-left:3px solid var(--tie);padding:10px 14px;color:var(--dim);font-size:13px;
       background:#1c1a15;border-radius:0 8px 8px 0;margin:16px 0}
 textarea{width:100%;height:150px;background:#0f1114;color:var(--fg);border:1px solid var(--line);
          border-radius:8px;padding:10px;font:12px/1.5 ui-monospace,Consolas,monospace}
 kbd{background:#232830;border:1px solid var(--line);border-radius:4px;padding:1px 6px;
     font:12px ui-monospace,Consolas,monospace}
</style></head><body><div class="wrap">
<h1>blind preference test</h1>
<p class="sub">同じ 1 本の歌を、2 つの系がそれぞれ変換した結果です。どちらがどの系かは表示されません。</p>

<div class="warn"><b>「target に似ているか」では選びません。</b>
話者類似度は客観指標で測り終わっています。ここで聴くのは<b>指標が拾えないもの</b>です
―― <b>自然さ</b>（声として不自然でないか）、<b>こもり / ざらつき</b>（高域の欠落、ノイズ、
金属的な響き）、<b>歌としての破綻</b>（音程の揺れ、子音の消失、途切れ）。<br>
<b>引き分けは空欄ではなく tie。</b> 集計が区別します（差が無かったこと自体が結果です）。
<code>key.json</code> は集計まで開かないでください。</div>

<div id="ctx"></div>

<div class="bar"><i id="fill"></i></div>
<div class="count"><span id="done"></span><span id="pos"></span></div>

<div class="card">
 <div class="pair" id="pair"></div>
 <div class="clip" id="clip"></div>
 <div class="row">
  <button class="play" id="pAB">A → B を続けて <span style="color:var(--dim)">(1)</span></button>
  <button class="play" id="pA">A だけ <span style="color:var(--dim)">(2)</span></button>
  <button class="play" id="pB">B だけ <span style="color:var(--dim)">(3)</span></button>
  <button id="stop">停止 <span style="color:var(--dim)">(0)</span></button>
 </div>
 <div class="row">
  <button class="vote" data-v="A">A が近い <span style="color:var(--dim)">(A)</span></button>
  <button class="vote" data-v="B">B が近い <span style="color:var(--dim)">(B)</span></button>
  <button class="vote tie" data-v="tie">引き分け <span style="color:var(--dim)">(T)</span></button>
 </div>
 <div class="nav">
  <button id="prev">← 前へ</button><button id="next">次へ →</button>
 </div>
 <p class="keys">キー: <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> 再生 / <kbd>4</kbd> 変換元 / <kbd>A</kbd><kbd>B</kbd><kbd>T</kbd> 投票
  （投票すると自動で次へ）/ <kbd>←</kbd><kbd>→</kbd> 移動</p>
</div>

<p class="keys" style="margin-top:0">「A → B」は <b>同じ歌の A の変換結果 → 無音 0.7 秒 → B の変換結果</b>を
1 本にしたものです。<b>音量は揃えてあります</b>（大きいほうを選ばないため）。</p>

<div class="grid" id="grid"></div>

<h2 style="font-size:16px;margin:28px 0 6px">書き出し</h2>
<p class="sub" style="margin:0 0 10px">全部埋めたら、この内容を <code>out/m5/blind/sheet.csv</code> へ上書きします。</p>
<div class="row" style="margin-top:0">
 <button id="dl">sheet.csv をダウンロード</button>
 <button id="copy">クリップボードへコピー</button>
 <button id="clear">投票を全消去</button>
</div>
<textarea id="csv" readonly></textarea>

<audio id="au"></audio>
</div><script>
const ROWS = __ROWS__;
const REFS = __REFS__;
const HEADER = __HEADER__;
const KEY = "leapsinger-blind-votes";
const au = document.getElementById("au");
let i = 0, votes = {};
try { votes = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { votes = {}; }
for (const r of ROWS) { if (r.vote && !votes[r.pair]) votes[r.pair] = r.vote; }

const $ = id => document.getElementById(id);
function save() { try { localStorage.setItem(KEY, JSON.stringify(votes)); } catch (e) {} }

function csv() {
  const nl = String.fromCharCode(10);
  const lines = [HEADER];
  for (const r of ROWS) lines.push(r.pair + "," + r.clip + "," + (votes[r.pair] || ""));
  return lines.join(nl) + nl;
}

function play(kind) {
  const r = ROWS[i];
  au.src = kind === "AB" ? r.ab : (kind === "A" ? r.a : r.b);
  au.currentTime = 0;
  au.play().catch(() => {});
  for (const b of document.querySelectorAll(".play")) b.classList.remove("on");
  $("p" + kind).classList.add("on");
}

function vote(v) {
  votes[ROWS[i].pair] = v; save(); render();
  setTimeout(() => { if (i < ROWS.length - 1) { i++; render(); } }, 150);
}

function render() {
  const r = ROWS[i];
  $("pair").textContent = r.pair;
  $("clip").textContent = r.clip;
  const n = ROWS.filter(x => votes[x.pair]).length;
  $("fill").style.width = (100 * n / ROWS.length) + "%";
  $("done").textContent = n + " / " + ROWS.length + " 記入済み";
  $("pos").textContent = (i + 1) + " ペア目";
  for (const b of document.querySelectorAll(".vote"))
    b.dataset.on = votes[r.pair] === b.dataset.v ? "1" : "0";
  $("prev").disabled = i === 0;
  $("next").disabled = i === ROWS.length - 1;
  const g = $("grid"); g.innerHTML = "";
  ROWS.forEach((x, k) => {
    const b = document.createElement("button");
    b.textContent = x.pair.replace("pair", "") + (votes[x.pair] ? " ●" : "");
    if (votes[x.pair]) b.className = "done";
    if (k === i) b.className += " cur";
    b.onclick = () => { i = k; render(); };
    g.appendChild(b);
  });
  const sb = document.getElementById("pSRC");
  if (sb) sb.disabled = !r.source;
  $("csv").value = csv();
}

// 参照（target 本人の録音）と変換元。**どちらの系の出力でもない**ので、聴いても
// A / B の正体は分かりません。「自然な歌声とはどう鳴るか」の基準として使います。
(function context() {
  const box = document.getElementById("ctx");
  if (!REFS.length) return;
  const row = document.createElement("div");
  row.className = "row";
  const cap = document.createElement("p");
  cap.className = "sub";
  cap.style.margin = "18px 0 0";
  cap.innerHTML = "<b>参照</b>（どちらの系の出力でもありません）";
  box.appendChild(cap);
  for (const r of REFS) {
    const b = document.createElement("button");
    b.textContent = r.label;
    b.onclick = () => { au.src = r.path; au.currentTime = 0; au.play().catch(() => {}); };
    row.appendChild(b);
  }
  const src = document.createElement("button");
  src.id = "pSRC";
  src.textContent = "このペアの変換元 (4)";
  src.onclick = () => playSource();
  row.appendChild(src);
  box.appendChild(row);
})();

function playSource() {
  const r = ROWS[i];
  if (!r.source) return;
  au.src = r.source; au.currentTime = 0; au.play().catch(() => {});
}

$("pAB").onclick = () => play("AB");
$("pA").onclick = () => play("A");
$("pB").onclick = () => play("B");
$("stop").onclick = () => au.pause();
$("prev").onclick = () => { if (i > 0) { i--; render(); } };
$("next").onclick = () => { if (i < ROWS.length - 1) { i++; render(); } };
for (const b of document.querySelectorAll(".vote")) b.onclick = () => vote(b.dataset.v);
$("dl").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv()], {type: "text/csv"}));
  a.download = "sheet.csv"; a.click();
};
$("copy").onclick = () => {
  const t = $("csv"); t.select(); t.setSelectionRange(0, 99999);
  navigator.clipboard.writeText(csv()).catch(() => document.execCommand("copy"));
  $("copy").textContent = "コピーしました";
  setTimeout(() => { $("copy").textContent = "クリップボードへコピー"; }, 1200);
};
$("clear").onclick = () => {
  if (confirm("投票を全部消します。よろしいですか？")) { votes = {}; save(); render(); }
};
addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  if (k === "1") play("AB"); else if (k === "2") play("A"); else if (k === "3") play("B");
  else if (k === "4") playSource();
  else if (k === "0") au.pause();
  else if (k === "a") vote("A"); else if (k === "b") vote("B"); else if (k === "t") vote("tie");
  else if (e.key === "ArrowLeft") { if (i > 0) { i--; render(); } }
  else if (e.key === "ArrowRight") { if (i < ROWS.length - 1) { i++; render(); } }
  else return;
  e.preventDefault();
});
render();
</script></body></html>
"""


def clip_tag(name: str) -> str:
    """変換結果のファイル名から clip の tag を取る。

    LeapSVC は `<name>__<tag>_converted.wav`、正規化した baseline は `<tag>_converted.wav`
    です。**どちらからも同じ tag が取れないと、2 系を対にできません。**
    """
    stem = str(name)
    if stem.endswith("_converted.wav"):
        stem = stem[: -len("_converted.wav")]
    return stem.split("__")[-1]


def find_clips(root: Path) -> dict[str, Path]:
    """`*_converted.wav` を tag -> path で返す。

    **source と上限は clip として数えません**（同じ tag で 3 本になり、対応が壊れます）。
    """
    return {clip_tag(p.name): p for p in sorted(Path(root).glob("*_converted.wav"))}


def assign_sides(clips: Sequence[str], *, systems: tuple[str, str],
                 seed: int) -> list[dict[str, str]]:
    """clip ごとに A / B の割り当てと提示順を決める。

    **clip の並びも混ぜます。** 曲順で聴くと後半に慣れが出るためです。
    """
    if len(systems) != 2 or systems[0] == systems[1]:
        raise ValueError(f"systems は異なる 2 つにしてください（{systems} が来ました）")
    rng = random.Random(seed)
    order = list(clips)
    rng.shuffle(order)
    rows = []
    for clip in order:
        first, second = (systems if rng.random() < 0.5 else (systems[1], systems[0]))
        rows.append({"clip": str(clip), "A": first, "B": second})
    return rows


def _p_sign_test(wins: int, n: int) -> float:
    """符号検定の両側 p 値（帰無仮説: 五分五分）。"""
    if n <= 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def tally(sheet: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """採点シートを system ごとの勝ち数へ直す。

    **side ではなく system で数えます**（A が常に同じ system とは限らないため）。
    **引き分けと未記入も数えます。**
    """
    wins: dict[str, int] = {}
    ties = missing = 0
    for row in sheet:
        vote = str(row.get("vote", "")).strip()
        if not vote:
            missing += 1
            continue
        if vote.lower() == "tie":
            ties += 1
            continue
        if vote not in ("A", "B"):
            raise ValueError(
                f"vote は A / B / tie / 空 のいずれかです（{vote!r} が来ました）。"
                "system 名を直接書くと、どちら側で聴いたのかが失われます")
        winner = str(row[vote])
        wins[winner] = wins.get(winner, 0) + 1

    decisive = sum(wins.values())
    names = sorted({str(r[k]) for r in sheet for k in ("A", "B") if k in r})
    for n in names:
        wins.setdefault(n, 0)
    top = max(wins.values()) if wins else 0
    return {
        "wins": wins, "ties": ties, "n_missing": missing,
        "n_voted": decisive + ties, "n_decisive": decisive,
        "p_two_sided": _p_sign_test(top, decisive),
        "note": ("符号検定は clip を独立とみなした参考値。評価者は 1 名なので、"
                 "報告では N=1 と clip 数を必ず併記する"),
    }


def _cmd_prepare(a) -> int:
    import json

    a_dir, b_dir = Path(a.a), Path(a.b)
    a_clips, b_clips = find_clips(a_dir), find_clips(b_dir)
    shared = sorted(set(a_clips) & set(b_clips))
    if not shared:
        sys.exit(f"共通の clip がありません（{a.a}: {len(a_clips)} / {a.b}: {len(b_clips)}）")
    print(f"[blind] 共通 clip {len(shared)} 本（{a.a_name} {len(a_clips)} / "
          f"{a.b_name} {len(b_clips)}）")

    rows = assign_sides(shared, systems=(a.a_name, a.b_name), seed=a.seed)
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    src = {a.a_name: a_clips, a.b_name: b_clips}
    import soundfile as sf

    key, sheet = [], [SHEET_HEADER]
    for i, row in enumerate(rows):
        pair = f"pair{i:02d}"
        # **同じ loudness 揃えを両側へ当てる**（事前登録した条件）。コピーでは駄目。
        wavs, srs = [], []
        for side in ("A", "B"):
            w, sr = sf.read(src[row[side]][row["clip"]], dtype="float32", always_2d=False)
            if w.ndim > 1:
                w = w.mean(axis=1)
            wavs.append(w)
            srs.append(sr)
        if srs[0] != srs[1]:
            sys.exit(f"{pair}: sample rate が違います（{srs}）。揃えてから比べること")
        matched = match_loudness(*wavs)
        for side, w in zip(("A", "B"), matched, strict=True):
            sf.write(out / "audio" / f"{pair}_{side}.wav", w, srs[0])
        # 続けて聴ける形（A -> 無音 -> B）。切り替えの手間を減らす。
        (out / "paired").mkdir(parents=True, exist_ok=True)
        sf.write(out / "paired" / f"{pair}_AB.wav", concat_pair(*matched, sr=srs[0]), srs[0])
        key.append({"pair": pair, **row})
        sheet.append(f"{pair},{row['clip']},")
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    (out / "sheet.csv").write_text("\n".join(sheet) + "\n", encoding="utf-8")
    print(f"  -> {out}/audio （**system 名は出ません**）")
    print(f"  -> {out}/paired （A -> 無音 -> B を 1 本にしたもの。続けて聴ける）")
    print(f"  -> {out}/sheet.csv （vote 列を埋める）")
    print(f"  -> {out}/key.json （集計まで見ないこと）")
    return 0


def _cmd_tally(a) -> int:
    import csv
    import json

    key = {r["pair"]: r for r in json.loads(Path(a.key).read_text(encoding="utf-8"))}
    sheet = []
    with open(a.sheet, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pair = (row.get("pair") or "").strip()
            if not pair or pair not in key:
                continue
            sheet.append({"clip": key[pair]["clip"], "A": key[pair]["A"], "B": key[pair]["B"],
                          "vote": (row.get("vote") or "").strip()})
    if not sheet:
        sys.exit("採点シートに有効な行がありません")

    rep = tally(sheet)
    rep["rows"] = sheet
    print("=== blind preference（N=1、非公式）===")
    for name, w in sorted(rep["wins"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:10s} {w:3d} 勝")
    print(f"  引き分け {rep['ties']} / 未記入 {rep['n_missing']} / 判定 {rep['n_decisive']}")
    print(f"  符号検定 p = {rep['p_two_sided']:.4f}（参考値）")
    print(f"\n  ** {rep['note']} **")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


def _cmd_page(a) -> int:
    import csv

    rows = []
    with open(a.sheet, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pair = (row.get("pair") or "").strip()
            if pair:
                rows.append({"pair": pair, "clip": (row.get("clip") or "").strip(),
                             "vote": (row.get("vote") or "").strip()})
    out = Path(a.out) if a.out else Path(a.sheet).parent / "listen.html"
    root = out.parent

    # 参照と変換元は **中立な名前で context/ へ複写**する。元の場所を直接指すと
    # パスに system 名が出て blind が崩れる（_check_path が拒否する）。
    refs = []
    if a.target_ref or a.source_dir:
        import soundfile as sf

        ctx = root / "context"
        ctx.mkdir(parents=True, exist_ok=True)

        def _copy(src_path, dst_name, level_of):
            """`level_of` の音量へ合わせて複写する（浮くと聴きにくいだけなので揃える）。"""
            w, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
            if w.ndim > 1:
                w = w.mean(axis=1)
            ref, _ = sf.read(str(level_of), dtype="float32", always_2d=False)
            if ref.ndim > 1:
                ref = ref.mean(axis=1)
            sf.write(ctx / dst_name, match_loudness(w, ref)[0], sr)
            return f"context/{dst_name}"

        if a.target_ref:
            level = root / "audio" / f"{rows[0]['pair']}_A.wav"
            refs.append({"label": a.target_label,
                         "path": _copy(a.target_ref, "target_ref.wav", level)})
        if a.source_dir:
            found = 0
            for r in rows:
                src = find_source(a.source_dir, r["clip"])
                if src is None:
                    continue
                level = root / "audio" / f"{r['pair']}_A.wav"
                r["source"] = _copy(src, f"{r['pair']}_source.wav", level)
                found += 1
            print(f"[blind] 変換元 {found} / {len(rows)} 本を context/ へ")

    out.write_text(listen_page(rows, references=refs), encoding="utf-8")
    print(f"[blind] {len(rows)} ペア -> {out}")
    print("  ブラウザで開いて投票し、書き出した sheet.csv で上書きしてから tally を回します")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prepare", help="聴く用のファイルと採点シートを作る")
    p1.add_argument("--a", required=True, help="片方の変換結果ディレクトリ")
    p1.add_argument("--b", required=True, help="もう片方")
    p1.add_argument("--a-name", default="A系")
    p1.add_argument("--b-name", default="B系")
    p1.add_argument("--out", required=True)
    p1.add_argument("--seed", type=int, default=0)
    p1.set_defaults(func=_cmd_prepare)

    p2 = sub.add_parser("tally", help="採点シートを集計する")
    p2.add_argument("--sheet", required=True)
    p2.add_argument("--key", required=True)
    p2.add_argument("--out", default=None)
    p2.set_defaults(func=_cmd_tally)

    p3 = sub.add_parser("page", help="ブラウザで聴いて投票するページを作る")
    p3.add_argument("--sheet", required=True)
    p3.add_argument("--out", default=None, help="既定は sheet.csv と同じ場所の listen.html")
    p3.add_argument("--target-ref", default=None,
                    help="target 話者本人の録音。**上限（*_vocoder_only.wav）は渡さないこと**")
    p3.add_argument("--target-label", default="target 本人の録音")
    p3.add_argument("--source-dir", default=None,
                    help="`<曲>__<clip>_source.wav` が並ぶディレクトリ（変換元。両系で同一）")
    p3.set_defaults(func=_cmd_page)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
