"""Build reports/results.html from the run logs. Offline: no API calls, no writes except the report.

    python -m evals.report [--out reports/results.html]

Every number is computed from runs/*_pipeline.jsonl (plus the baselines and answer keys) with the
same code the eval harness uses: evals.scoring, evals.regression, evals.readme.run_cost,
evals.blind_spot and evals.self_correction. Each section names the run log(s) it uses.

Run labels:
  fault-injection  header has inject_faults > 0 (an experiment; values were corrupted on purpose)
  superseded       a later run has the same split, model, prompts, self-correct and fault settings
                   on the same base commit (the commit recorded by its first eval record); a run
                   logged before the eval harness existed (no eval record) is superseded by any
                   later run with the same settings
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from pathlib import Path

from evals import regression
from evals.blind_spot import outcomes as blind_spot_outcomes
from evals.readme import run_cost
from evals.scoring import compute_metrics, load_log, split_of
from evals.self_correction import analyse as analyse_self_correction
from validation.validate import EXCEPTION, validate_records

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "reports" / "results.html"
SPLITS = ("main", "hard")
TYPES = ("CAS", "SOI", "LETTER")


# --------------------------------------------------------------------------- runs

def _config(h: dict) -> tuple:
    fs = h.get("fault_summary") or {}
    return (split_of(h), h.get("model"), json.dumps(h.get("prompt_versions"), sort_keys=True),
            bool(h.get("self_correct")), h.get("inject_faults") or 0, fs.get("seed"))


def load_runs(runs_dir: Path = RUNS) -> list[dict]:
    runs = []
    for path in sorted(runs_dir.glob("*_pipeline.jsonl")):
        header, records, evals = load_log(path)
        split = split_of(header)
        stored = [(r.get("validation") or {}).get("status") for r in records]
        statuses = stored if all(stored) else [v.status for v in validate_records(records)]
        cost = run_cost(header, records)
        runs.append({
            "path": path, "name": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "header": header, "records": records,
            "split": split, "model": header.get("model"), "time": header.get("timestamp", ""),
            "fault": bool(header.get("inject_faults")), "self_correct": bool(header.get("self_correct")),
            "base_commit": (evals[0]["git"]["commit"] if evals else None),
            # webhook uploads have no answer key: no accuracy, only exceptions and cost
            "metrics": compute_metrics(records, split=split) if split in SPLITS else None, "cost": cost,
            "exception_rate": sum(s == EXCEPTION for s in statuses) / len(statuses) if statuses else 0.0,
        })
    for i, r in enumerate(runs):
        if r["split"] not in SPLITS:  # webhook sessions add documents; none replaces another
            r["superseded_by"] = None
            continue
        later = [x for x in runs[i + 1:] if _config(x["header"]) == _config(r["header"])]
        r["superseded_by"] = next((x["name"] for x in later
                                   if r["base_commit"] is None or x["base_commit"] == r["base_commit"]), None)
    return runs


def label(r: dict) -> str:
    tags = (["webhook uploads"] if r["split"] == "webhook" else []) + \
           ([f"fault-injection ({r['header'].get('inject_faults')} faults)"] if r["fault"] else []) + \
           (["self-correct"] if r["self_correct"] and not r["fault"] else []) + \
           ([f"superseded by {Path(r['superseded_by']).name[:16]}"] if r["superseded_by"] else [])
    return ", ".join(tags)


def by_name(runs: list[dict], path: Path | None) -> dict | None:
    return next((r for r in runs if path and r["path"] == path), None)


# --------------------------------------------------------------------------- formatting

def esc(x) -> str:
    return html.escape(str(x))


def pct(x, dp=1) -> str:
    return "—" if x is None else f"{100 * x:.{dp}f}%"


def usd(x, dp=4) -> str:
    return "—" if x is None else f"${x:,.{dp}f}"


def num(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if isinstance(x, float):  # display only: 3.7299999999999995 -> 3.73
        return f"{x:,.4f}".rstrip("0").rstrip(".")
    return f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else str(x)


def when(r: dict) -> str:
    try:
        return dt.datetime.fromisoformat(r["time"]).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return r["time"]


def run_tag(r: dict) -> str:
    extra = label(r)
    return (f"<code>{esc(r['name'])}</code> · {esc(r['split'])} split · {esc(r['model'])} · {esc(when(r))}"
            + (f" · {esc(extra)}" if extra else ""))


def uses(*rs: dict | None) -> str:
    items = "".join(f"<li>{run_tag(r)}</li>" for r in rs if r)
    return f'<div class="uses"><span>Run{"s" if len([r for r in rs if r]) > 1 else ""} used</span><ul>{items}</ul></div>'


def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def type_counts(m: dict, t: str) -> tuple[int, int]:
    rows = [v for k, v in m["field_counts"].items() if k.startswith(t + ".")]
    return sum(r[0] for r in rows), sum(r[1] for r in rows)


# --------------------------------------------------------------------------- sections

def gate_runs(runs) -> dict[str, dict | None]:
    return {s: by_name(runs, regression.gate_log(s)) for s in SPLITS}


def sec_headline(runs) -> str:
    g = gate_runs(runs)
    rows = []
    for s in SPLITS:
        r = g[s]
        if not r:
            continue
        m, e = r["metrics"], r["metrics"]["exceptions"]
        rows.append([f"<b>{s}</b>", num(m["n_docs"]), esc(r["model"]), pct(m["router_accuracy"]),
                     pct(m["field_accuracy_overall"]), f"{e['precision']:.2f}", f"{e['recall']:.2f}",
                     usd(r["cost"]["usd"], 2)])
    return (uses(*g.values()) +
            table(["Split", "Docs", "Model", "Router accuracy", "Field accuracy", "Exception precision",
                   "Exception recall", "Cost per run"], rows, "num") +
            '<p class="note">Each split uses the run the regression gate checks: its newest run by the baseline '
            "model, excluding fault-injection runs. Exception precision and recall are over CAS, SOI and LETTER "
            "documents; cost is at list prices in <code>config.PRICING_PER_MTOK</code>.</p>")


def sec_gate(runs) -> str:
    out = []
    g = gate_runs(runs)
    for s in SPLITS:
        r = g[s]
        base_path = regression.BASELINES[s]
        if not r or not base_path.exists():
            out.append(f"<h3>{s}</h3><p>No baseline or no gate run yet.</p>")
            continue
        base = regression.load_baseline(base_path)
        problems = regression.find_regressions(r["metrics"], base)
        badge = ('<span class="badge pass">&#10003; PASS</span>' if not problems
                 else '<span class="badge fail">&#10007; FAIL</span>')
        bm = base["metrics"]
        drops = [[esc(f), pct(bm["field_accuracy"][f]), pct(r["metrics"]["field_accuracy"].get(f)),
                  f"{100 * (r['metrics']['field_accuracy'].get(f, 0) - bm['field_accuracy'][f]):+.1f} pts"]
                 for f in bm["field_accuracy"] if r["metrics"]["field_accuracy"].get(f, 0) < bm["field_accuracy"][f]]
        detail = ("<ul class='problems'>" + "".join(f"<li>{esc(p)}</li>" for p in problems) + "</ul>") if problems \
            else f"<p>No field dropped. {len(bm['field_accuracy'])} fields compared; router accuracy " \
                 f"{pct(bm['router_accuracy'])} &rarr; {pct(r['metrics']['router_accuracy'])}; exception recall " \
                 f"{bm['exceptions']['recall']:.2f} &rarr; {r['metrics']['exceptions']['recall']:.2f}.</p>"
        out.append(f"<h3>{s} split {badge}</h3>" + uses(r) +
                   f"<p class='note'>Baseline <code>{esc(base_path.relative_to(ROOT))}</code> from "
                   f"<code>{esc(base['source_log'])}</code> at commit <code>{esc((base['git']['commit'] or '')[:8])}"
                   f"</code>. Fails if any field drops more than {regression.FIELD_DROP_POINTS:g} points, or router "
                   "accuracy or exception recall drops at all.</p>" + detail +
                   (table(["Field", "Baseline", "Latest", "Change"], drops, "num") if drops else ""))
    return "".join(out)


def sec_fields(runs) -> str:
    g = gate_runs(runs)
    out = [uses(*g.values())]
    for t in TYPES:
        fields = list(dict.fromkeys(k.split(".", 1)[1] for s in SPLITS if g[s]
                                    for k in g[s]["metrics"]["field_counts"] if k.startswith(t + ".")))
        rows = []
        for f in fields:
            row = [f"<code>{esc(f)}</code>"]
            for s in SPLITS:
                c = g[s]["metrics"]["field_counts"].get(f"{t}.{f}") if g[s] else None
                row.append(f"{pct(c[0] / c[1])} <span class='muted'>({c[0]}/{c[1]})</span>" if c else "—")
            rows.append(row)
        totals = ["<b>All fields</b>"]
        for s in SPLITS:
            ok, n = type_counts(g[s]["metrics"], t) if g[s] else (0, 0)
            totals.append(f"<b>{pct(ok / n) if n else '—'}</b> <span class='muted'>({ok}/{n})</span>")
        out.append(f"<h3>{t}</h3>" + table(["Field", "main split", "hard split"], rows + [totals], "num"))
    out.append("<p class='note'>List fields are scored per item, matched by name: <code>holdings.cost</code> "
               "counts every holding's cost; <code>holdings[present]</code> counts holdings found at all. "
               "<code>commentary_summary</code> and <code>confidence</code> are not scored.</p>")
    return "".join(out)


def sec_models(runs) -> str:
    latest: dict[str, dict] = {}
    for r in runs:
        if r["split"] == "hard" and not r["fault"] and not r["superseded_by"]:
            latest[r["model"]] = r
    if len(latest) < 2:
        return "<p>Fewer than two models have a hard-split run.</p>"
    rs = [latest[m] for m in sorted(latest, key=lambda m: ("haiku" in m, m))]
    rows = [["Router accuracy", *[pct(r["metrics"]["router_accuracy"]) for r in rs]]]
    for t in TYPES:
        rows.append([f"{t} fields", *[(lambda c: f"{pct(c[0] / c[1])} ({c[0]}/{c[1]})")(type_counts(r["metrics"], t))
                                      for r in rs]])
    rows += [
        ["Exception precision / recall", *[f"{r['metrics']['exceptions']['precision']:.2f} / "
                                           f"{r['metrics']['exceptions']['recall']:.2f}" for r in rs]],
        ["Tokens in / out", *[f"{r['cost']['tin']:,} / {r['cost']['tout']:,}" for r in rs]],
        ["Cost per run", *[usd(r["cost"]["usd"]) for r in rs]],
        ["Cost per doc", *[usd(r["cost"]["usd"] / r["cost"]["n"], 5) for r in rs]],
        ["Latency per doc, mean / max", *[f"{r['cost']['doc_mean']:.1f} s / {r['cost']['doc_max']:.1f} s" for r in rs]],
    ]
    cheaper = rs[0]["cost"]["usd"] / rs[1]["cost"]["usd"] if rs[1]["cost"]["usd"] else None
    return (uses(*rs) + table(["Hard split", *[f"<code>{esc(r['model'])}</code>" for r in rs]], rows, "num") +
            (f"<p class='note'>Same documents, same v1 prompts. {esc(rs[1]['model'])} costs "
             f"{1 / cheaper:.0%} of {esc(rs[0]['model'])} per run.</p>" if cheaper else ""))


def fault_pairs(runs) -> dict[str, tuple[dict, dict]]:
    pairs = {}
    for s in SPLITS:
        fr = [r for r in runs if r["split"] == s and r["fault"] and r["self_correct"] and not r["superseded_by"]]
        if len(fr) >= 2:
            pairs[s] = (fr[0], fr[-1])
    return pairs


def sec_self_correction(runs) -> str:
    pairs = fault_pairs(runs)
    if not pairs:
        return "<p>No self-correction fault-injection runs.</p>"
    out = [uses(*[r for p in pairs.values() for r in p]),
           "<p class='note'>Before = the earliest current fault-injection run of a split (Phase 9 checks); after = the "
           "latest (Phase 9b checks added). Both use the same seed and fault count, so the same faults are injected. "
           "Recovered = the final value equals the answer key; flagged = still wrong but sent to review; "
           "undetected = still wrong and accepted (the blind spot).</p>"]
    summary = []
    for s, (b, a) in pairs.items():
        ob, bm = blind_spot_outcomes(b["path"])
        oa, am = blind_spot_outcomes(a["path"])
        sb, sa = analyse_self_correction(b["path"]), analyse_self_correction(a["path"])
        for name, o, sc, m, run in (("before", ob, sb, bm, b), ("after", oa, sa, am, a)):
            n = len(o)
            c = {v: sum(x["verdict"] == v for x in o.values()) for v in ("recovered", "flagged", "undetected")}
            summary.append([f"<b>{s}</b>", name, f"<code>{esc(run['path'].name[:16])}</code>",
                            f"{c['recovered']}/{n}", f"{c['flagged']}/{n}", f"{c['undetected']}/{n}",
                            num(len(sc["fudges"])), num(len(sc["accepted_on_broken"])),
                            f"{m['precision']:.2f} / {m['recall']:.2f}",
                            f"{usd(sc['retry_usd'])} (+{100 * sc['retry_usd'] / sc['first_usd']:.0f}%)"])
    out.append(table(["Split", "Checks", "Run", "Recovered", "Flagged", "Undetected", "Fudges rejected",
                      "Changes accepted on broken docs", "Exception P / R (docs without a fault)",
                      "Extra cost of retries"], summary, "num"))
    for s, (b, a) in pairs.items():
        ob, _ = blind_spot_outcomes(b["path"])
        oa, _ = blind_spot_outcomes(a["path"])
        rows = []
        for k in sorted(set(ob) | set(oa)):
            bo, ao = ob.get(k, {}).get("verdict", "—"), oa.get(k, {}).get("verdict", "—")
            closed = bo == "undetected" and ao != "undetected"
            rows.append([esc(k[0]), f"<code>{esc(k[1])}</code>", esc(k[2]), f"{esc(num(_f(k[3])))} &rarr; {esc(num(_f(k[4])))}",
                         f"<span class='v {bo}'>{bo}</span>", f"<span class='v {ao}'>{ao}</span>",
                         esc(", ".join(oa.get(k, {}).get("caught_by", [])) or "—") + (" <b>closed</b>" if closed else "")])
        out.append(f"<h3>{s} split, per fault</h3>" +
                   table(["Doc", "Field", "Fault", "Value", "Before", "After", "Caught by (after)"], rows))
    return "".join(out)


def _f(s: str):
    try:
        return float(s)
    except ValueError:
        return s


def sec_exceptions(runs) -> str:
    g = gate_runs(runs)
    out = [uses(*g.values()),
           "<p class='note'>Re-validated from each run's records with the current rules (the same check the gate and "
           "the scorer use). One row per failed check.</p>"]
    for s in SPLITS:
        r = g[s]
        if not r:
            continue
        rows = []
        for v in validate_records(r["records"]):
            for i in v.issues:
                rows.append([esc(v.doc_id), esc(v.doc_type or "—"), f"<code>{esc(i.rule)}</code>", esc(num(i.expected)),
                             esc(num(i.actual)), esc(num(i.difference)), esc(i.reason)])
        out.append(f"<h3>{s} split · {len({x[0] for x in rows})} documents flagged</h3>" +
                   table(["Doc", "Type", "Rule", "Expected", "Actual", "Difference", "Reason"], rows))
    wh_runs = [r for r in runs if r["split"] == "webhook"]
    if wh_runs:
        latest = {}
        for r in wh_runs:
            for rec in r["records"]:
                latest[Path(rec["file"]).stem] = rec
        rows = []
        for v in validate_records(list(latest.values())):
            rec = latest[v.doc_id]
            name = (rec.get("webhook") or {}).get("filename") or "—"
            for i in v.issues:
                rows.append([f"<code>{esc(v.doc_id[:12])}</code> ({esc(name)})", esc(v.doc_type or "—"),
                             f"<code>{esc(i.rule)}</code>", esc(num(i.expected)), esc(num(i.actual)),
                             esc(num(i.difference)), esc(i.reason)])
        out.append(uses(*wh_runs) +
                   f"<h3>webhook uploads · {len({x[0] for x in rows})} of {len(latest)} documents flagged</h3>" +
                   (table(["Doc (SHA-256 prefix)", "Type", "Rule", "Expected", "Actual", "Difference", "Reason"], rows)
                    if rows else "<p>None flagged.</p>"))
    return "".join(out)


# --------------------------------------------------------------------------- run history chart

def chart(runs) -> str:
    W, left, right, ph, gap, top = 760, 64, 16, 120, 60, 40
    n = len(runs)
    xs = [left + (i + 0.5) * (W - left - right) / n for i in range(n)]
    panels = [
        ("Field accuracy", lambda r: r["metrics"]["field_accuracy_overall"] * 100 if r["metrics"] else None,
         lambda v: f"{v:.1f}%", None),
        ("Exception rate (share of docs flagged)", lambda r: r["exception_rate"] * 100, lambda v: f"{v:.0f}%", 0.0),
        ("Cost per doc (list price)", lambda r: (r["cost"]["usd"] or 0) / max(1, r["cost"]["n"]),
         lambda v: f"${v:.3f}", 0.0),
    ]
    color = {"main": "var(--s1)", "hard": "var(--s2)", "webhook": "var(--s3)"}
    parts = []
    y0 = top
    for title, get, fmt, floor in panels:
        vals = [get(r) for r in runs]
        known = [v for v in vals if v is not None]
        lo = floor if floor is not None else max(0.0, min(known) - max(0.5, (max(known) - min(known)) * 0.2))
        hi = max(known) * 1.15 if floor is not None else 100.0
        lo = min(lo, hi - 1e-9)
        Y = lambda v: y0 + ph - (v - lo) / (hi - lo) * ph  # noqa: E731
        parts.append(f'<text x="{left - 56}" y="{y0 - 14}" class="pt">{esc(title)}</text>')
        for k in range(3):
            tv = lo + (hi - lo) * k / 2
            ty = Y(tv)
            parts.append(f'<line x1="{left}" x2="{W - right}" y1="{ty:.1f}" y2="{ty:.1f}" class="grid"/>'
                         f'<text x="{left - 8}" y="{ty + 4:.1f}" class="tick" text-anchor="end">{esc(fmt(tv))}</text>')
        for r, x, v in zip(runs, xs, vals):
            if v is None:  # no answer key for this run (webhook uploads)
                continue
            faded = " faded" if r["superseded_by"] else ""
            fill = "var(--surface)" if r["fault"] else color[r["split"]]
            tip = (f"{r['path'].name[:16]} · {r['split']} · {r['model']}\n{title}: {fmt(v)}"
                   + (f"\n{label(r)}" if label(r) else ""))
            parts.append(f'<g class="pt-mark{faded}"><title>{esc(tip)}</title>'
                         f'<circle cx="{x:.1f}" cy="{Y(v):.1f}" r="11" fill="transparent"/>'
                         f'<circle cx="{x:.1f}" cy="{Y(v):.1f}" r="5" fill="{fill}" stroke="{color[r["split"]]}" '
                         f'stroke-width="2"/></g>')
        y0 += ph + gap
    axis_y = y0 - gap + 18
    models = [r["model"] for r in runs]
    usual = max(set(models), key=models.count)
    for i, (r, x) in enumerate(zip(runs, xs)):
        t = dt.datetime.fromisoformat(r["time"])
        flag = "F" if r["fault"] else ("S" if r["superseded_by"] else ("W" if r["split"] == "webhook" else ""))
        if r["model"] != usual:  # name the odd model out on the axis, e.g. the Haiku run
            flag = (flag + " " if flag else "") + r["model"].split("-")[1].capitalize()
        parts.append(f'<text x="{x:.1f}" y="{axis_y}" class="tick" text-anchor="middle">{t:%H:%M}</text>'
                     f'<text x="{x:.1f}" y="{axis_y + 14}" class="tick" text-anchor="middle">{esc(flag)}</text>')
    parts.append(f'<text x="{left}" y="{axis_y + 32}" class="tick">Runs in time order ({esc(runs[0]["time"][:10])} '
                 f'to {esc(runs[-1]["time"][:10])}, UTC). F = fault-injection, S = superseded, W = webhook uploads; model {esc(usual)} unless named.</text>')
    H = axis_y + 44
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Field accuracy, exception rate and cost per doc '
            f'for every logged run">{"".join(parts)}</svg>')


def sec_history(runs) -> str:
    legend = ('<div class="legend"><span><i class="dot" style="background:var(--s1)"></i>main split</span>'
              '<span><i class="dot" style="background:var(--s2)"></i>hard split</span>'
              '<span><i class="dot" style="background:var(--s3)"></i>webhook uploads (no accuracy: no answer key)</span>'
              '<span><i class="dot ring"></i>fault-injection run (hollow)</span>'
              '<span><i class="dot" style="background:var(--s1);opacity:.35"></i>superseded run (faded)</span></div>')
    rows = [[esc(when(r)), f"<code>{esc(r['path'].name[:16])}</code>", esc(r["split"]), esc(r["model"]),
             pct(r["metrics"]["field_accuracy_overall"]) if r["metrics"] else "n/a", pct(r["exception_rate"], 0),
             usd((r["cost"]["usd"] or 0) / max(1, r["cost"]["n"])), esc(label(r) or "—")] for r in runs]
    return (uses(*runs) + legend + f'<figure class="chart">{chart(runs)}</figure>' +
            "<p class='note'>Fault-injection runs corrupt extracted values on purpose before the retry, so their "
            "accuracy and exception rate measure the self-correction loop, not ordinary extraction. Hover a point for "
            "its run.</p>" +
            "<details><summary>Table view of the chart</summary>" +
            table(["Started", "Run", "Split", "Model", "Field accuracy", "Exception rate", "Cost per doc", "Label"],
                  rows, "num") + "</details>")


# --------------------------------------------------------------------------- page

CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f6f5f2;--ink:#0b0b0b;--ink2:#52514e;--rule:#e2e0da;
--grid:#ebe9e4;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0a7a0a;--bad:#c62f2e;--goodbg:#e5f4e4;--badbg:#fbebeb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--page:#121211;
--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835;--grid:#2c2c2a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4cc04c;--bad:#ef7b7b;
--goodbg:#173017;--badbg:#3a1b1b}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:0 0 12px}h3{font-size:15px;margin:20px 0 8px}
section{background:var(--surface);border:1px solid var(--rule);border-radius:12px;padding:20px 22px;margin:18px 0}
.sub{color:var(--ink2);margin:0}.note{color:var(--ink2);font-size:13px}.muted{color:var(--ink2)}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--ink2);font-size:12.5px}table.num td{font-variant-numeric:tabular-nums}
.uses{font-size:12.5px;color:var(--ink2);border-left:3px solid var(--rule);padding:2px 0 2px 10px;margin:0 0 12px}
.uses span{font-weight:600}.uses ul{margin:2px 0 0;padding-left:18px}
.badge{display:inline-block;font-size:12px;font-weight:700;padding:2px 10px;border-radius:999px;margin-left:8px}
.badge.pass{color:var(--good);background:var(--goodbg)}.badge.fail{color:var(--bad);background:var(--badbg)}
.problems{color:var(--bad)}.v{font-weight:600}.v.recovered{color:var(--good)}.v.undetected{color:var(--bad)}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink2);margin:4px 0 8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.dot.ring{border:2px solid var(--ink2);background:transparent}
.chart{margin:0}.chart svg{width:100%;height:auto;display:block}
svg .grid{stroke:var(--grid);stroke-width:1}svg .tick{fill:var(--ink2);font-size:11.5px}
svg .pt{fill:var(--ink);font-size:13px;font-weight:600}svg .faded{opacity:.35}
details summary{cursor:pointer;color:var(--ink2);font-size:13px;margin-top:8px}
"""


def build(runs_dir: Path = RUNS) -> str:
    runs = load_runs(runs_dir)
    sections = [
        ("Headline", sec_headline(runs)),
        ("Regression gate", sec_gate(runs)),
        ("Per-field accuracy", sec_fields(runs)),
        ("Sonnet vs. Haiku", sec_models(runs)),
        ("Self-correction", sec_self_correction(runs)),
        ("Exceptions queue", sec_exceptions(runs)),
        ("Run history", sec_history(runs)),
    ]
    body = "".join(f'<section id="s{i}"><h2>{i}. {esc(t)}</h2>{c}</section>' for i, (t, c) in enumerate(sections, 1))
    newest = runs[-1]
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Fund Doc Agent Results</title><style>{CSS}</style></head><body><main>"
            "<h1>Fund document extraction: results</h1>"
            f"<p class='sub'>Built offline from {len(runs)} pipeline run logs in <code>runs/</code>, newest "
            f"<code>{esc(newest['path'].name)}</code> ({esc(when(newest))}). Synthetic documents only. Regenerate with "
            "<code>python -m evals.report</code>.</p>" + body + "</main></body></html>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build())
    print(f"Wrote {out.relative_to(ROOT) if out.resolve().is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
