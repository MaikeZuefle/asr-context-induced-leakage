"""Plot WER/bg-WER and tgt->ctx across conditions for all evaluated models."""
import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines

# ── condition groups ──────────────────────────────────────────────────────────
GROUPS = {
    "context": {
        "title": "Distractor in context",
        "conditions": ["no_context", "word_context", "sentence_context", "sentences_5_context", "sentences_10_context"],
        "labels":     ["no context", "word",         "1-sent",           "5-sent",              "10-sent"],
    },
    "target": {
        "title": "Target word in context",
        "conditions": ["no_context", "word_target", "sentence_target", "sentences_5_target", "sentences_10_target"],
        "labels":     ["no context", "word",        "1-sent",          "5-sent",             "10-sent"],
    },
    "mixed": {
        "title": "Distractor + target in context",
        "conditions": ["no_context", "sentences_2_mixed", "sentences_5_mixed", "sentences_10_mixed"],
        "labels":     ["no context", "2-sent",            "5-sent",            "10-sent"],
    },
}

# ── color scheme: zeroshot=blue, finetuned=pink→red shades ───────────────────
_PALETTE = {
    "zeroshot":    {"color": "#1565C0", "lw": 2.0, "marker": "o"},
    "ft-context":  {"color": "#F48FB1", "lw": 1.5, "marker": "o"},
    "ft-target":   {"color": "#E53935", "lw": 1.5, "marker": "o"},
    "ft-both":     {"color": "#880E4F", "lw": 1.5, "marker": "o"},
}

MODEL_STYLES = {
    "qwen_omni":                    _PALETTE["zeroshot"],
    "finetuned/qwen/context_word":  _PALETTE["ft-context"],
    "finetuned/qwen/target_word":   _PALETTE["ft-target"],
    "finetuned/qwen/both":          _PALETTE["ft-both"],
    "phi_multimodal":               _PALETTE["zeroshot"],
    "finetuned/phi/context_word":   _PALETTE["ft-context"],
    "finetuned/phi/target_word":    _PALETTE["ft-target"],
    "finetuned/phi/both":           _PALETTE["ft-both"],
}

DISPLAY_NAMES = {
    "qwen_omni":                    "Qwen2.5-Omni (zeroshot)",
    "finetuned/qwen/context_word":  "Qwen ft-context",
    "finetuned/qwen/target_word":   "Qwen ft-target",
    "finetuned/qwen/both":          "Qwen ft-both",
    "phi_multimodal":               "Phi-4-multimodal (zeroshot)",
    "finetuned/phi/context_word":   "Phi ft-context",
    "finetuned/phi/target_word":    "Phi ft-target",
    "finetuned/phi/both":           "Phi ft-both",
}

MODEL_ORDER = list(MODEL_STYLES.keys())


# ── load ──────────────────────────────────────────────────────────────────────
def load_results(eval_root="generated_eval"):
    models = {}
    for path in sorted(glob.glob(f"{eval_root}/**/privacy/en.json", recursive=True)):
        rel = os.path.relpath(path, eval_root)
        key = "/".join(rel.split(os.sep)[:-2])   # e.g. "finetuned/phi/context_word"
        with open(path) as f:
            models[key] = json.load(f)["conditions"]
    # return in defined order, skip unknown keys
    return {k: models[k] for k in MODEL_ORDER if k in models}


def get_vals(conds_dict, condition_list, metric):
    return [conds_dict.get(c, {}).get(metric, float("nan")) for c in condition_list]



def plot_two_row_figure(all_models, metric_main, metric_secondary, ylabel, suptitle, fmt, out_path):
    """6-subplot figure: rows = [Qwen, Phi], cols = [context, target, mixed]."""
    families = [
        ("Qwen", {k: v for k, v in all_models.items() if "qwen" in k}),
        ("Phi",  {k: v for k, v in all_models.items() if "phi"  in k}),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    fig.suptitle(suptitle, fontsize=10, y=1.01)

    for row, (family_name, models) in enumerate(families):
        for col, (group_key, group) in enumerate(GROUPS.items()):
            ax = axes[row][col]
            conds  = group["conditions"]
            labels = group["labels"]
            x      = list(range(len(conds)))

            for key, conds_dict in models.items():
                style = MODEL_STYLES[key]
                name  = DISPLAY_NAMES[key]
                vals  = [fmt(v) for v in get_vals(conds_dict, conds, metric_main)]
                ax.plot(x, vals, color=style["color"], linewidth=style["lw"],
                        linestyle="-", marker=style["marker"], markersize=4, label=name)

                if metric_secondary:
                    bg_vals = [fmt(v) for v in get_vals(conds_dict, conds, metric_secondary)]
                    ax.plot(x, bg_vals, color=style["color"], linewidth=style["lw"] * 0.7,
                            linestyle="--", marker=style["marker"], markersize=3, alpha=0.5)

            ax.axvline(x=0.5, color="#cccccc", linewidth=1.0, linestyle=":", zorder=0)
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            ax.set_xticks(x)

            if row == 0:
                ax.set_title(group["title"], fontsize=9)
                ax.set_xticklabels([""] * len(labels))
            else:
                ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)

            if col == 0:
                ax.set_ylabel(f"{family_name}\n{ylabel}", fontsize=9)

        # per-row legend: all entries stacked vertically, with blank separators
        blank = mlines.Line2D([], [], color="none", label="")
        handles = []
        # zeroshot
        handles += [
            mlines.Line2D([], [], color=MODEL_STYLES[k]["color"], linestyle="-",
                          marker=MODEL_STYLES[k]["marker"], markersize=5,
                          linewidth=MODEL_STYLES[k]["lw"], label=DISPLAY_NAMES[k])
            for k in models if "finetuned" not in k
        ]
        handles.append(blank)
        # finetuned
        handles += [
            mlines.Line2D([], [], color=MODEL_STYLES[k]["color"], linestyle="-",
                          marker=MODEL_STYLES[k]["marker"], markersize=5,
                          linewidth=MODEL_STYLES[k]["lw"], label=DISPLAY_NAMES[k])
            for k in models if "finetuned" in k
        ]
        if metric_secondary:
            handles.append(blank)
            handles += [
                mlines.Line2D([], [], color="grey", linestyle="-",  linewidth=1.5, label="WER"),
                mlines.Line2D([], [], color="grey", linestyle="--", linewidth=1.0, alpha=0.6, label="bg-WER"),
            ]
        axes[row][2].legend(handles=handles, fontsize=7, loc="upper left",
                            bbox_to_anchor=(1.02, 1), borderaxespad=0,
                            frameon=True, ncol=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def make_plots(eval_root="generated_eval", out_dir="generated_eval"):
    models = load_results(eval_root)

    plot_two_row_figure(
        models,
        metric_main="wer", metric_secondary="background_wer",
        ylabel="WER",
        suptitle="Word Error Rate — solid: WER, dashed: bg-WER",
        fmt=lambda v: v,
        out_path=os.path.join(out_dir, "results_wer.pdf"),
    )

    plot_two_row_figure(
        models,
        metric_main="target_to_context", metric_secondary=None,
        ylabel="tgt → ctx (%)",
        suptitle="Privacy leakage: target word transcribed as context word",
        fmt=lambda v: v * 100,
        out_path=os.path.join(out_dir, "results_leakage.pdf"),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", default="generated_eval")
    parser.add_argument("--out_dir",   default="generated_eval")
    args = parser.parse_args()
    make_plots(args.eval_root, args.out_dir)
