"""
Reporting. Every artifact carries its claim boundary; the boundary is checked in code.

Reports are written so that a reader who skims only the headline cannot come away with a
stronger claim than the data support. The disposition line always names what is excluded
in the same breath as what is licensed.
"""

from __future__ import annotations

from .validate import enforce_claim_boundary, bridge_report


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)):
        return str(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:
        return "nan"
    if f in (float("inf"), float("-inf")):
        return "inf"
    return f"{f:.{nd}g}"


def render_report(result: dict) -> str:
    m = result["meta"]
    d = result["design_assessment"]
    s = result["structure"]
    v = result["validation"]
    disp = result["disposition"]

    L = []
    A = L.append
    A(f"# Ei run report — {m['experiment']}")
    A("")
    A(f"- **Pre-registration hash**: `{m['preregistration_hash'][:16]}…`")
    A(f"- **Source**: {m['source']}  |  **Read-out**: `{m['readout']}`  |  d={m['hidden_dim']}")
    A(f"- **Seed**: {m['seed']}  |  **Splits**: {m['split_sizes']}  |  {m['wallclock_s']}s")
    A("")
    A("> **Claim ceiling.** " + m["claim_ceiling"])
    A("")

    A("## 1. Design assessment (pre-data)")
    A("")
    A(f"- Coverage: **{d['coverage']*100:.1f}%**")
    A(f"- Disposition: **{d['disposition']}**")
    if d["blocking"]:
        A("- Blocking gaps:")
        for b in d["blocking"]:
            A(f"  - {b}")
    else:
        A("- Blocking gaps: none")
    A("")
    A("| check | passed | weight | rationale |")
    A("|---|---|---|---|")
    for c in d["checks"]:
        A(f"| `{c['key']}` | {'PASS' if c['passed'] else 'FAIL'} | {c['weight']} | {c['detail']} |")
    A("")

    A("## 2. Structure")
    A("")
    g = s["gate_report"]
    A(f"- Modes fitted: {g['n_modes_fitted']}  |  candidates after gates: **{g['n_candidates']}**")
    A(f"- Variance captured by reduction: {_fmt(g['variance_captured_by_reduction'])}")
    A(f"- Marchenko–Pastur noise-floor energy share: {_fmt(g['noise_floor_energy_share'])}")
    A(f"- Gates: {g['gates']}")
    A("")
    A("| mode | persistence \\|λ\\| | half-life | freq | energy | consistency | SNR |")
    A("|---|---|---|---|---|---|---|")
    for md in s["modes"][:8]:
        A(f"| {md['index']} | {_fmt(md['persistence'])} | {_fmt(md['half_life'])} | "
          f"{_fmt(md['frequency'])} | {_fmt(md['energy_share'])} | {_fmt(md['consistency'])} | "
          f"{_fmt(md['snr'])} |")
    A("")
    A(f"*{g['note']}*")
    A("")

    A("## 3. Validation")
    A("")
    A("### 3.1 Null ladder")
    A("")
    if v["null_tests"]:
        A("| null | statistic | comparison | p | decision |")
        A("|---|---|---|---|---|")
        for k, t in v["null_tests"].items():
            comp = (f"null 95% [{_fmt(t.get('ci_low'))}, {_fmt(t.get('ci_high'))}]"
                    if "ci_low" in t else
                    f"baseline={_fmt(t.get('baseline_max_over_seeds', t.get('mimic_statistic')))}")
            A(f"| `{k}` | {_fmt(t.get('statistic'))} | {comp} | {_fmt(t.get('p_value'))} | "
              f"**{t.get('decision','n/a')}** |")
        A("")
        for k, t in v["null_tests"].items():
            if t.get("note"):
                A(f"- `{k}`: {t['note']}")
    else:
        A("No null tests were run. **No structural claim is licensed.**")
    A("")

    if v["bootstrap"]:
        b = v["bootstrap"]
        A("### 3.2 Interval estimate")
        A("")
        A(f"- Estimate {_fmt(b['estimate'])}, {int(b['level']*100)}% BCa CI "
          f"[{_fmt(b['ci_low'])}, {_fmt(b['ci_high'])}] over {b['n_boot']} item-cluster resamples")
        A("")

    if v["equivalence"]:
        e = v["equivalence"]
        A("### 3.3 Equivalence (evidence of absence)")
        A("")
        A(f"- Excess over phase-scramble null: {_fmt(e['estimate'])}, "
          f"{int(e['ci_level']*100)}% CI [{_fmt(e['ci_low'])}, {_fmt(e['ci_high'])}], "
          f"SESOI ±{_fmt(e['margin'])}")
        A(f"- Statistically equivalent to zero: **{'yes' if e['equivalent'] else 'no'}**")
        A(f"- {e['note']}")
        A("")

    if v["alignment"]:
        A("### 3.4 Cross-model alignment")
        A("")
        for tag, al in v["alignment"].items():
            A(f"**{tag}** — {al.get('_note','')}")
            A("")
            A("| metric | score | null mean | excess | z | p | reading |")
            A("|---|---|---|---|---|---|---|")
            for mt in ("cka", "svcca", "procrustes"):
                r = al.get(mt)
                if not r:
                    continue
                A(f"| {mt} | {_fmt(r['score'])} | {_fmt(r['null_mean'])} | {_fmt(r['excess'])} | "
                  f"{_fmt(r['z'])} | {_fmt(r['p_value'])} | {r['interpretation']} |")
            sens = al.get("_svcca_sensitivity", {})
            if sens:
                A("")
                A(f"- SVCCA threshold sensitivity: range {_fmt(sens.get('range'))}, "
                  f"stable={_fmt(sens.get('stable'))}")
            A("")

    if v["causal"]:
        c = v["causal"]
        A("### 3.5 Causal battery")
        A("")
        if "primary" in c:
            p = c["primary"]
            A(f"- Intact {_fmt(p['intact'])} → lesioned {_fmt(p['lesioned'])} "
              f"(effect {_fmt(p['effect'])})")
            A(f"- Matched-random ablation: {_fmt(p['matched_random'])} → "
              f"**specificity {_fmt(p['specificity'])}**")
            A(f"- Off-target effect: {_fmt(p['off_target_effect'])}")
            A(f"- Restoration fraction: **{_fmt(p['restoration_fraction'])}**")
            dr = c.get("dose_response", {})
            A(f"- Dose–response ρ={_fmt(dr.get('rho'))}, monotone={_fmt(dr.get('monotone'))}")
        vd = c.get("verdict", {})
        A(f"- Causal verdict: **{'PASS' if vd.get('pass') else 'FAIL'}**")
        for k, ok in (vd.get("checks") or {}).items():
            A(f"  - {k}: {'PASS' if ok else 'FAIL'}")
        for f in vd.get("failures", []):
            A(f"  - ✗ {f}")
        A("")

    A("## 4. Disposition")
    A("")
    A(f"### `{disp['disposition']}`")
    A("")
    A(disp["rationale"])
    A("")
    A(f"**Licensed claim.** {disp['licensed_claim']}")
    A("")
    A(f"**Excluded claim.** {disp['excluded_claim']}")
    A("")
    if disp.get("failing_checks"):
        A("**Failing checks.**")
        for f in disp["failing_checks"]:
            A(f"- {f}")
        A("")
    if disp.get("bridge_note"):
        A("**Bridging principle (assumed, not tested).**")
        A("")
        A(disp["bridge_note"])
        A("")

    text = "\n".join(L)
    enforce_claim_boundary(text, context="report.md")
    return text


def render_interpretation(result: dict) -> str:
    disp = result["disposition"]
    v = result["validation"]
    L = []
    A = L.append
    A("# How to read these artifacts")
    A("")
    A("## Files")
    A("")
    A("| file | what it is | how to read it |")
    A("|---|---|---|")
    A("| `results.json` | machine-readable record of every test | the authoritative artifact; "
      "the report is a rendering of it |")
    A("| `report.md` | human-readable run report | read section 3 before section 4; the "
      "disposition is only as good as the ladder above it |")
    A("| `preregistration.txt` | hash of the result-affecting config | if this differs from the "
      "registered hash, the run is EXPLORATORY regardless of what it says |")
    A("| `INTERPRETATION.md` | this file | read first |")
    A("")
    A("## Reading order that avoids over-claiming")
    A("")
    A("1. **Check the pre-registration hash.** Mismatch ⇒ exploratory. Stop treating any p-value "
      "as confirmatory.")
    A("2. **Check the design coverage and blocking gaps.** A design missing the mimic control or "
      "the rescue arm cannot support a mechanistic claim however the numbers landed.")
    A("3. **Check the phase-scramble null specifically.** Beating temporal shuffle alone is close "
      "to meaningless: autocorrelated signals always look persistent.")
    A("4. **Check specificity, not effect size.** `effect` compares to nothing. `specificity` "
      "compares to ablating a matched random subspace of the same rank, which is the number that "
      "supports a localisation claim.")
    A("5. **Check restoration fraction.** RF < 0.6 ⇒ the lesion broke something the subspace does "
      "not carry. RF > 1.25 ⇒ the rescue added information; that is a failed control, not a "
      "strong result.")
    A("6. **Read the excluded claim** in section 4 before quoting the licensed one.")
    A("")
    A("## What each outcome means")
    A("")
    A("| statistic | positive evidence | inconclusive | evidence of absence |")
    A("|---|---|---|---|")
    A("| persistence vs phase-scramble | p < α_primary **and** excess > SESOI | p < α but excess < "
      "SESOI (significant, trivial) | TOST equivalent to 0 within ±SESOI |")
    A("| untrained baseline | trained ≫ max over init seeds | overlapping | trained ≤ baseline ⇒ "
      "architectural artifact |")
    A("| output-matched mimic | mimic statistic < 50% of target | 50–90% | mimic ≈ target ⇒ the "
      "measure tracks output, not mechanism |")
    A("| Procrustes cross-arch | p < 0.05 **and** excess > 0.05 | p < 0.05, excess small | not "
      "distinguishable from broken-pairing null |")
    A("| restoration fraction | 0.6 ≤ RF ≤ 1.25 with specificity > 0 | RF in (0, 0.6) | RF ≈ 0 with "
      "tight CI ⇒ subspace is not load-bearing |")
    A("")
    A("## The ceiling, restated")
    A("")
    A("The strongest disposition this framework can emit is "
      "`computational_correlate_supported`. It means: a persistent, mimic-resistant, causally "
      "load-bearing, cross-architecture structure implements a **functionally defined** construct. "
      "It is a fact about computation. Moving from there to a claim about experience requires a "
      "bridging principle that no current theory has established, and which this framework treats "
      "as a declared input rather than a finding.")
    A("")
    b = bridge_report("recurrent_processing")
    A("### The bridge this run assumed, stated plainly")
    A("")
    A(f"- **Principle**: {b['claim']}")
    A(f"- **Status**: {b['status']}")
    A(f"- **A pass shows**: {b['what_a_pass_shows']}")
    A(f"- **A pass does not show**: {b['what_a_pass_does_not_show']}")
    A(f"- **Principal objection**: {b['principal_objection']}")
    A("")
    A(f"**This run's disposition**: `{disp['disposition']}`")
    A("")
    A(f"**Excluded**: {disp['excluded_claim']}")
    A("")
    if v.get("causal", {}).get("verdict", {}).get("failures"):
        A("### Targeted next steps (from this run's failures)")
        A("")
        for f in v["causal"]["verdict"]["failures"]:
            A(f"- {f}")
        A("")
    text = "\n".join(L)
    enforce_claim_boundary(text, context="INTERPRETATION.md")
    return text
