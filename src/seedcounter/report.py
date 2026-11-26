"""Output tables (CSV), the HTML review report and results.json."""
from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

from .model import RunResult, to_dict
from .settings_info import fingerprint
from .summary import REVIEW_RANK, image_summary, seed_class

IMAGE_FIELDS = ["image", "pods", "seeds", "auto_seeds", "change", "green", "orange", "red", "pods_no", "pods_maybe", "pods_yes",
                "needs_review", "flag_source", "reviewed", "note", "annotated_image"]
POD_FIELDS = ["image", "pod", "seed_count", "auto_count", "change", "top_row", "bottom_row", "green", "orange", "red",
              "pod_confidence", "needs_review", "flag_source", "reviewed", "reason"]
SEED_FIELDS = ["image", "pod", "seed_id", "row", "x_px", "y_px", "area_px", "major_px", "minor_px",
               "confidence", "class", "status", "source"]


def summaries(run: RunResult, cfg: dict) -> list[dict]:
    return [image_summary(img, cfg) for img in run.images]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    # utf-8-sig: Excel and Google Sheets both read the file correctly on Windows and Linux
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_totals(sums: list[dict]) -> dict:
    n_review = sum(s["needs_review"] != "no" for s in sums)
    n_err = sum(s["needs_review"] == "error" for s in sums)
    label = f"TOTAL ({len(sums)} images processed; {n_review} need review"
    label += f", {n_err} could not be processed)" if n_err else ")"
    tot = {k: sum(s[k] for s in sums) for k in ("pods", "seeds", "auto_seeds", "change", "green", "orange", "red",
                                                 "pods_no", "pods_maybe", "pods_yes")}
    n_rev_done = sum(s.get("reviewed") == "yes" for s in sums)
    if n_rev_done:
        label = label[:-1] + f"; {n_rev_done} checked by a reviewer)"
    checked = [r for s in sums if s.get("reviewed") == "yes" for r in s["pod_rows"]]
    acc = None
    if checked:
        final = sum(r["seed_count"] for r in checked)
        off = sum(abs(r["change"]) for r in checked)
        acc = {"pods": len(checked), "seeds": final, "abs_change": off,
               "per_pod": off / len(checked), "rel": off / final if final else 0.0}
    return {"image": label, **tot, "accuracy_on_checked": acc, "needs_review": "", "reviewed": "", "note": "", "annotated_image": "",
            "images_processed": len(sums), "images_needing_review": n_review, "images_failed": n_err}


def write_tables(run: RunResult, cfg: dict, table_dir: Path, sums: list[dict]) -> list[Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    out = cfg["output"]
    written = []
    if out.get("run_summary_csv", True):
        p = table_dir / "run_summary.csv"
        _write_csv(p, sums + [run_totals(sums)], IMAGE_FIELDS)
        written.append(p)
    pod_rows = [{"image": s["image"], **r} for s in sums for r in s["pod_rows"]]
    if out.get("pods_csv", True):
        p = table_dir / "pods.csv"
        _write_csv(p, pod_rows, POD_FIELDS)
        written.append(p)
    if out.get("report_outline_csv", True):
        p = table_dir / "report_outline.csv"
        tot = run_totals(sums)
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["level", "image", "pod", "seeds", "auto_count", "change", "top_row", "bottom_row", "green",
                        "orange", "red", "confidence", "needs_review", "reviewed", "reason"])
            w.writerow(["RUN", f"{tot['images_processed']} images processed", "", tot["seeds"], tot["auto_seeds"],
                        tot["change"], "", "", tot["green"],
                        tot["orange"], tot["red"], "", f"{tot['images_needing_review']} images need review", "", ""])
            for s in sums:
                w.writerow(["IMAGE", s["image"], f"{s['pods']} pods", s["seeds"], s["auto_seeds"], s["change"], "", "",
                            s["green"], s["orange"],
                            s["red"], "", s["needs_review"], s["reviewed"], s["note"]])
                for r in s["pod_rows"]:
                    w.writerow(["   pod", "", r["pod"], r["seed_count"], r["auto_count"], r["change"], r["top_row"], r["bottom_row"], r["green"],
                                r["orange"], r["red"], r["pod_confidence"], r["needs_review"], r["reviewed"], r["reason"]])
        written.append(p)
    if out.get("seeds_csv", False):
        rows = []
        for img in run.images:
            for pod in img.pods:
                for sd in pod.seeds:
                    rows.append({"image": img.image, "pod": pod.index, "seed_id": sd.id, "row": sd.row,
                                 "x_px": sd.centroid[0], "y_px": sd.centroid[1], "area_px": sd.area_px,
                                 "major_px": sd.major_px, "minor_px": sd.minor_px, "confidence": sd.confidence,
                                 "class": seed_class(sd, cfg) if sd.status == "active" else "rejected",
                                 "status": sd.status, "source": sd.source})
        p = table_dir / "seeds.csv"
        _write_csv(p, rows, SEED_FIELDS)
        written.append(p)
    return written


def write_json(run: RunResult, cfg: dict, path: Path, sums: list[dict]) -> Path | None:
    mode = cfg["output"].get("results_json", "run")
    if mode in ("off", False, None):
        return None
    tot = run_totals(sums)
    data = {
        "run": {"date": run.date, "version": run.version, "settings_fingerprint": fingerprint(run.config or cfg),
                "config_file": run.config_file, "input_dir": run.input_dir,
                "review_threshold": cfg["confidence"]["review_threshold"],
                "reject_threshold": cfg["confidence"]["reject_threshold"],
                "images_processed": tot["images_processed"], "images_needing_review": tot["images_needing_review"],
                "images_failed": tot["images_failed"], "total_seeds": tot["seeds"], "total_automatic": tot["auto_seeds"],
                "accuracy_on_checked": tot["accuracy_on_checked"], "settings": run.config},
        "images": [{k: v for k, v in s.items() if k != "pod_rows"} for s in sums],
    }
    if mode == "full":
        data["results"] = [to_dict(img) for img in run.images]
        data["summaries"] = sums
    path.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    return path


# ------------------------------------------------------------------ HTML
def _esc(x) -> str:
    return html.escape(str(x))


def _bar(g, o, r) -> str:
    n = max(g + o + r, 1)
    return (f'<span class="bar" role="img" aria-label="{g} green, {o} orange, {r} red">'
            f'<i class="g" style="width:{g / n * 100:.1f}%"></i><i class="o" style="width:{o / n * 100:.1f}%"></i>'
            f'<i class="r" style="width:{r / n * 100:.1f}%"></i></span><span class="gor">{g} / {o} / {r}</span>')


def _chg(c) -> str:
    return f' <span class="chg" title="change from the automatic count">({c:+d})</span>' if c else ""


def _chip(s) -> str:
    return f'<span class="chip {_esc(s)}">{_esc(s)}</span>'


def _anchor(name: str) -> str:
    return "img-" + "".join(c if c.isalnum() else "-" for c in Path(name).stem)


CSS = """
:root{--bg:#f5f6f2;--surface:#fff;--ink:#1f2620;--muted:#5d665e;--line:#dfe2da;--pod:#d8c27a;
--g:#2f8a3e;--o:#d97a16;--r:#c43b30;--link:#2d5f8a;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#171a17;--surface:#1f231f;--ink:#e7eae3;--muted:#a2aa9f;
--line:#343a33;--pod:#b9a35c;--g:#5cbf6a;--o:#f0a04b;--r:#ee6d62;--link:#8fb8dc;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI","Noto Sans","Helvetica Neue",Arial,sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1240px;margin:0 auto;padding:28px 20px 60px}
header{border-left:6px solid var(--pod);padding:4px 0 4px 16px;margin-bottom:22px}
h1{font-size:26px;margin:0 0 4px;font-weight:650}.sub{color:var(--muted);margin:0}
.verdict{font-size:19px;margin:18px 0 12px}.verdict b{font-size:28px}
a{color:var(--link)}
table{border-collapse:collapse;width:100%;background:var(--surface)}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:13px}
td.n{text-align:right}td.strong{font-weight:700}td.reason{white-space:normal;color:var(--muted);min-width:220px}
.scroll{overflow-x:auto}
.chip{display:inline-block;min-width:54px;text-align:center;border-radius:999px;padding:1px 10px;font-size:13px;font-weight:600;border:1.5px solid currentColor}
.chip.no{color:var(--g)}.chip.maybe{color:var(--o)}.chip.yes,.chip.error{color:#fff;background:var(--r);border-color:var(--r)}
.bar{display:inline-flex;width:120px;height:10px;border-radius:3px;overflow:hidden;vertical-align:middle;background:var(--line)}
.bar i{display:block;height:100%}.bar .g{background:var(--g)}.bar .o{background:var(--o)}.bar .r{background:var(--r)}
.gor{margin-left:8px;color:var(--muted);font-size:13px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.dot.green{background:var(--g)}.dot.orange{background:var(--o)}.dot.red{background:var(--r)}
.controls{margin:26px 0 10px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
details.img{background:var(--surface);border:1px solid var(--line);border-radius:6px;margin:10px 0}
details.img[data-status=yes],details.img[data-status=error]{border-left:5px solid var(--r)}
details.img[data-status=maybe]{border-left:5px solid var(--o)}details.img[data-status=no]{border-left:5px solid var(--g)}
summary{cursor:pointer;padding:12px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
summary .name{font-weight:650}summary .meta{color:var(--muted)}
.body{padding:0 16px 16px;display:grid;gap:16px}
.thumb img{max-width:100%;width:900px;border:1px solid var(--line);border-radius:4px;display:block}
.note{color:var(--muted);margin:0}
button.exp{font:inherit;background:none;border:0;color:var(--link);cursor:pointer;padding:0}
button.exp::before{content:"+ "}button.exp[aria-expanded=true]::before{content:"\\2212  "}
tr.seeds>td{background:var(--bg);padding:8px 10px 12px 28px}
table.inner{width:auto;min-width:60%}table.inner td,table.inner th{padding:4px 10px}
:focus-visible{outline:2px solid var(--link);outline-offset:2px}
.legend{color:var(--muted);font-size:13px;margin-top:6px}
.rev{font-size:12px;color:var(--muted);margin-left:6px}
.chg{font-weight:400;color:var(--muted);font-size:12px}
"""

JS = """
document.querySelectorAll("button.exp[aria-controls]").forEach(b=>b.addEventListener("click",()=>{
 const row=document.getElementById(b.getAttribute("aria-controls"));const open=b.getAttribute("aria-expanded")!=="true";
 b.setAttribute("aria-expanded",open);row.hidden=!open;}));
document.getElementById("only").addEventListener("change",e=>{
 document.querySelectorAll("details.img").forEach(d=>d.hidden=e.target.checked&&d.dataset.status==="no");});
document.getElementById("all").addEventListener("click",e=>{const open=e.target.getAttribute("aria-expanded")!=="true";
 document.querySelectorAll("details.img").forEach(d=>d.open=open);e.target.setAttribute("aria-expanded",open);
 e.target.textContent=open?"Close all images":"Open all images";});
"""


def _accuracy(tot) -> str:
    a = tot.get("accuracy_on_checked")
    if not a:
        return ""
    return (f'<p class="sub">On the {a["pods"]} pods of checked images, reviewers changed the automatic counts by '
            f'{a["abs_change"]} seeds in total ({a["per_pod"]:.1f} per pod, {a["rel"] * 100:.1f} % of {a["seeds"]} seeds).</p>')


def write_html(run: RunResult, cfg: dict, path: Path, sums: list[dict], thumbs: dict[str, bytes]) -> Path:
    c = cfg["confidence"]
    tot = run_totals(sums)
    overview, sections = [], []
    for img, s in zip(run.images, sums):
        a = _anchor(img.image)
        flagged = f'{s["pods_maybe"]} maybe, {s["pods_yes"]} yes' if s["needs_review"] != "error" else "not processed"
        overview.append(
            f'<tr data-status="{_esc(s["needs_review"])}"><td><a href="#{a}">{_esc(img.image)}</a></td>'
            f'<td>{_chip(s["needs_review"])}{" <span class=rev>flag set by reviewer</span>" if s.get("flag_source") == "reviewer" else ""}{" <span class=rev>reviewed</span>" if s.get("reviewed") == "yes" else ""}</td><td class="n">{s["pods"]}</td><td class="n">{s["seeds"]}{_chg(s["change"])}</td>'
            f'<td>{_bar(s["green"], s["orange"], s["red"])}</td><td class="n">{flagged}</td></tr>')
        pod_html = []
        pods_by_index = {p.index: p for p in img.pods}
        for r in s["pod_rows"]:
            pod = pods_by_index[r["pod"]]
            sid = f"{a}-pod{pod.index}"
            seed_rows = "".join(
                f'<tr><td>{_esc(sd.id)}</td><td>{sd.row}</td><td class="n">{sd.centroid[0]:.0f}</td>'
                f'<td class="n">{sd.centroid[1]:.0f}</td><td class="n">{sd.area_px:.0f}</td>'
                f'<td class="n">{sd.confidence:.2f}</td><td><span class="dot {seed_class(sd, cfg)}"></span>{seed_class(sd, cfg)}</td></tr>'
                for sd in pod.active_seeds())
            pod_html.append(
                f'<tr data-status="{r["needs_review"]}"><td><button class="exp" aria-expanded="false" aria-controls="{sid}">Pod {pod.index}</button></td>'
                f'<td class="n strong">{r["seed_count"]}{_chg(r["change"])}</td><td class="n">{r["top_row"]} / {r["bottom_row"]}</td>'
                f'<td>{_bar(r["green"], r["orange"], r["red"])}</td><td class="n">{r["pod_confidence"]:.2f}</td>'
                f'<td>{_chip(r["needs_review"])}</td><td class="reason">{_esc(r["reason"])}</td></tr>'
                f'<tr class="seeds" id="{sid}" hidden><td colspan="7"><div class="scroll"><table class="inner"><thead><tr>'
                f'<th>Seed</th><th>Row</th><th>x (px)</th><th>y (px)</th><th>Area (px)</th><th>Confidence</th><th>Class</th>'
                f'</tr></thead><tbody>{seed_rows}</tbody></table></div></td></tr>')
        thumb = ""
        if img.image in thumbs:
            b64 = base64.b64encode(thumbs[img.image]).decode()
            link = _esc(s["annotated_image"]) if s["annotated_image"] else "#"
            thumb = f'<a class="thumb" href="{link}"><img alt="Annotated {_esc(img.image)}" src="data:image/jpeg;base64,{b64}"></a>'
        note = f'<p class="note">{_esc(s["note"])}</p>' if s["note"] else ""
        table = ("" if not pod_html else
                 '<div class="scroll"><table><thead><tr><th>Pod</th><th>Seeds</th><th>Top / bottom</th>'
                 '<th>Green / orange / red</th><th>Pod confidence</th><th>Needs review</th><th>Reason</th></tr></thead>'
                 f'<tbody>{"".join(pod_html)}</tbody></table></div>')
        sections.append(
            f'<details class="img" id="{a}" data-status="{_esc(s["needs_review"])}" {"open" if s["needs_review"] != "no" else ""}>'
            f'<summary><span class="name">{_esc(img.image)}</span>{_chip(s["needs_review"])}'
            f'<span class="meta">{s["pods"]} pods, {s["seeds"]} seeds</span></summary>'
            f'<div class="body">{note}{thumb}{table}</div></details>')
    n_rev = tot["images_needing_review"]
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seed count report</title><style>{CSS}</style></head><body><main>
<header><h1>Seed count report</h1>
<p class="sub">Run {_esc(run.date)}, {_esc(Path(run.config_file).name)}, review threshold {c["review_threshold"]:.2f},
reject threshold {c["reject_threshold"]:.2f}, input {_esc(run.input_dir)}<br>
Program version {_esc(run.version)}, settings fingerprint <b>{_esc(fingerprint(run.config or cfg))}</b>
(the same fingerprint means identical counting)</p></header>
<p class="verdict"><b>{n_rev} of {len(sums)}</b> images need review</p>{_accuracy(tot)}
<div class="scroll"><table><thead><tr><th>Image</th><th>Needs review</th><th>Pods</th><th>Seeds</th>
<th>Green / orange / red seeds</th><th>Flagged pods</th></tr></thead><tbody>{"".join(overview)}</tbody></table></div>
<p class="legend">Seed colours: green = confident, orange = check, red = likely wrong. All colours are counted.
Needs review comes from pod confidence: no at {c["review_threshold"]:.2f} or above, maybe between {c["reject_threshold"]:.2f}
and {c["review_threshold"]:.2f}, yes below {c["reject_threshold"]:.2f}. An image takes its worst pod.
Magenta outlines in the images mark dark areas without a seed border (possible missed seeds).</p>
<div class="controls"><label><input type="checkbox" id="only"> Show only images that need review</label>
<button class="exp" id="all" aria-expanded="false">Open all images</button></div>
{"".join(sections)}
</main><script>{JS}</script></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path


def sort_key(s: dict) -> int:
    return -REVIEW_RANK.get(s["needs_review"], 3)
