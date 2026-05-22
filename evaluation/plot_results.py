import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import numpy as np

# condition groups
GROUPS = {
    "context": {
        "title": "Distractor in context (target word in speech)",
        "conditions": ["no_context", "word_context", "sentence_context", "sentences_5_context", "sentences_10_context"],
        "labels":     ["no context", "word",         "1-sent",           "5-sent",              "10-sent"],
    },
    "target": {
        "title": "Target word in context (target word in speech)",
        "conditions": ["no_context", "word_target", "sentence_target", "sentences_5_target", "sentences_10_target"],
        "labels":     ["no context", "word",        "1-sent",          "5-sent",             "10-sent"],
    },
    "mixed": {
        "title": "Distractor + target in context (target word in speech)",
        "conditions": ["no_context", "sentences_2_mixed", "sentences_5_mixed", "sentences_10_mixed"],
        "labels":     ["no context", "2-sent",            "5-sent",            "10-sent"],
    },
}

# color scheme: zeroshot=blue, eval-data ft=pink→red, fleurs-context ft=light→dark green,
# combined (eval-data + fleurs-context) ft=light→dark purple
_PALETTE = {
    "zeroshot":                  {"color": "#1565C0", "lw": 2.0, "marker": "o"},
    "ft-context":                {"color": "#F48FB1", "lw": 1.5, "marker": "o"},
    "ft-target":                 {"color": "#E53935", "lw": 1.5, "marker": "o"},
    "ft-both":                   {"color": "#880E4F", "lw": 1.5, "marker": "o"},
    "ft-fleurs-1":               {"color": "#C8E6C9", "lw": 1.5, "marker": "s"},
    "ft-fleurs-5":               {"color": "#66BB6A", "lw": 1.5, "marker": "s"},
    "ft-fleurs-10":              {"color": "#1B5E20", "lw": 1.5, "marker": "s"},
    "ft-fleurs-mixed":           {"color": "#00897B", "lw": 1.5, "marker": "s"},
    "ft-combined-context":       {"color": "#CE93D8", "lw": 1.5, "marker": "^"},
    "ft-combined-target":        {"color": "#8E24AA", "lw": 1.5, "marker": "^"},
    "ft-combined-both":          {"color": "#4A148C", "lw": 1.5, "marker": "^"},
}

MODEL_STYLES = {
    # zero-shot
    "qwen_omni":                                       _PALETTE["zeroshot"],
    "phi_multimodal":                                  _PALETTE["zeroshot"],
    # FLEURS fine-tuned
    "qwen/context_word":                               _PALETTE["ft-context"],
    "qwen/target_word":                                _PALETTE["ft-target"],
    "qwen/both":                                       _PALETTE["ft-both"],
    "qwen/fleurs_context_1":                           _PALETTE["ft-fleurs-1"],
    "qwen/fleurs_context_5":                           _PALETTE["ft-fleurs-5"],
    "qwen/fleurs_context_10":                          _PALETTE["ft-fleurs-10"],
    "qwen/fleurs_context_mixed":                       _PALETTE["ft-fleurs-mixed"],
    "qwen/context_word_fleurs_mixed":                  _PALETTE["ft-combined-context"],
    "qwen/target_word_fleurs_mixed":                   _PALETTE["ft-combined-target"],
    "qwen/both_fleurs_mixed":                          _PALETTE["ft-combined-both"],
    "phi/context_word":                                _PALETTE["ft-context"],
    "phi/target_word":                                 _PALETTE["ft-target"],
    "phi/both":                                        _PALETTE["ft-both"],
    "phi/fleurs_context_1":                            _PALETTE["ft-fleurs-1"],
    "phi/fleurs_context_5":                            _PALETTE["ft-fleurs-5"],
    "phi/fleurs_context_10":                           _PALETTE["ft-fleurs-10"],
    "phi/fleurs_context_mixed":                        _PALETTE["ft-fleurs-mixed"],
    "phi/context_word_fleurs_mixed":                   _PALETTE["ft-combined-context"],
    "phi/target_word_fleurs_mixed":                    _PALETTE["ft-combined-target"],
    "phi/both_fleurs_mixed":                           _PALETTE["ft-combined-both"],
    # ACL6060 fine-tuned
    "qwen/acl6060_context_word":                       _PALETTE["ft-context"],
    "qwen/acl6060_target_word":                        _PALETTE["ft-target"],
    "qwen/acl6060_both":                               _PALETTE["ft-both"],
    "qwen/acl6060_context_word_fleurs_mixed":          _PALETTE["ft-combined-context"],
    "qwen/acl6060_target_word_fleurs_mixed":           _PALETTE["ft-combined-target"],
    "qwen/acl6060_both_fleurs_mixed":                  _PALETTE["ft-combined-both"],
    "phi/acl6060_context_word":                        _PALETTE["ft-context"],
    "phi/acl6060_target_word":                         _PALETTE["ft-target"],
    "phi/acl6060_both":                                _PALETTE["ft-both"],
    "phi/acl6060_context_word_fleurs_mixed":           _PALETTE["ft-combined-context"],
    "phi/acl6060_target_word_fleurs_mixed":            _PALETTE["ft-combined-target"],
    "phi/acl6060_both_fleurs_mixed":                   _PALETTE["ft-combined-both"],
    # VoxPopuli fine-tuned
    "qwen/voxpopuli_context_word":                     _PALETTE["ft-context"],
    "qwen/voxpopuli_target_word":                      _PALETTE["ft-target"],
    "qwen/voxpopuli_both":                             _PALETTE["ft-both"],
    "qwen/voxpopuli_context_word_fleurs_mixed":        _PALETTE["ft-combined-context"],
    "qwen/voxpopuli_target_word_fleurs_mixed":         _PALETTE["ft-combined-target"],
    "qwen/voxpopuli_both_fleurs_mixed":                _PALETTE["ft-combined-both"],
    "phi/voxpopuli_context_word":                      _PALETTE["ft-context"],
    "phi/voxpopuli_target_word":                       _PALETTE["ft-target"],
    "phi/voxpopuli_both":                              _PALETTE["ft-both"],
    "phi/voxpopuli_context_word_fleurs_mixed":         _PALETTE["ft-combined-context"],
    "phi/voxpopuli_target_word_fleurs_mixed":          _PALETTE["ft-combined-target"],
    "phi/voxpopuli_both_fleurs_mixed":                 _PALETTE["ft-combined-both"],
}

DISPLAY_NAMES = {
    "qwen_omni":                                       "Qwen2.5-Omni (zeroshot)",
    "phi_multimodal":                                  "Phi-4-multimodal (zeroshot)",
    "qwen/context_word":                               "Qwen ft-distractor",
    "qwen/target_word":                                "Qwen ft-target",
    "qwen/both":                                       "Qwen ft-both",
    "qwen/fleurs_context_1":                           "Qwen ft-fleurs-1sent",
    "qwen/fleurs_context_5":                           "Qwen ft-fleurs-5sent",
    "qwen/fleurs_context_10":                          "Qwen ft-fleurs-10sent",
    "qwen/fleurs_context_mixed":                       "Qwen ft-fleurs-mixed",
    "qwen/context_word_fleurs_mixed":                  "Qwen ft-distractor+fleurs",
    "qwen/target_word_fleurs_mixed":                   "Qwen ft-target+fleurs",
    "qwen/both_fleurs_mixed":                          "Qwen ft-both+fleurs",
    "phi/context_word":                                "Phi ft-distractor",
    "phi/target_word":                                 "Phi ft-target",
    "phi/both":                                        "Phi ft-both",
    "phi/fleurs_context_1":                            "Phi ft-fleurs-1sent",
    "phi/fleurs_context_5":                            "Phi ft-fleurs-5sent",
    "phi/fleurs_context_10":                           "Phi ft-fleurs-10sent",
    "phi/fleurs_context_mixed":                        "Phi ft-fleurs-mixed",
    "phi/context_word_fleurs_mixed":                   "Phi ft-distractor+fleurs",
    "phi/target_word_fleurs_mixed":                    "Phi ft-target+fleurs",
    "phi/both_fleurs_mixed":                           "Phi ft-both+fleurs",
    "qwen/acl6060_context_word":                       "Qwen ft-distractor (ACL)",
    "qwen/acl6060_target_word":                        "Qwen ft-target (ACL)",
    "qwen/acl6060_both":                               "Qwen ft-both (ACL)",
    "qwen/acl6060_context_word_fleurs_mixed":          "Qwen ft-distractor+fleurs (ACL)",
    "qwen/acl6060_target_word_fleurs_mixed":           "Qwen ft-target+fleurs (ACL)",
    "qwen/acl6060_both_fleurs_mixed":                  "Qwen ft-both+fleurs (ACL)",
    "phi/acl6060_context_word":                        "Phi ft-distractor (ACL)",
    "phi/acl6060_target_word":                         "Phi ft-target (ACL)",
    "phi/acl6060_both":                                "Phi ft-both (ACL)",
    "phi/acl6060_context_word_fleurs_mixed":           "Phi ft-distractor+fleurs (ACL)",
    "phi/acl6060_target_word_fleurs_mixed":            "Phi ft-target+fleurs (ACL)",
    "phi/acl6060_both_fleurs_mixed":                   "Phi ft-both+fleurs (ACL)",
    "qwen/voxpopuli_context_word":                     "Qwen ft-distractor (VP)",
    "qwen/voxpopuli_target_word":                      "Qwen ft-target (VP)",
    "qwen/voxpopuli_both":                             "Qwen ft-both (VP)",
    "qwen/voxpopuli_context_word_fleurs_mixed":        "Qwen ft-distractor+fleurs (VP)",
    "qwen/voxpopuli_target_word_fleurs_mixed":         "Qwen ft-target+fleurs (VP)",
    "qwen/voxpopuli_both_fleurs_mixed":                "Qwen ft-both+fleurs (VP)",
    "phi/voxpopuli_context_word":                      "Phi ft-distractor (VP)",
    "phi/voxpopuli_target_word":                       "Phi ft-target (VP)",
    "phi/voxpopuli_both":                              "Phi ft-both (VP)",
    "phi/voxpopuli_context_word_fleurs_mixed":         "Phi ft-distractor+fleurs (VP)",
    "phi/voxpopuli_target_word_fleurs_mixed":          "Phi ft-target+fleurs (VP)",
    "phi/voxpopuli_both_fleurs_mixed":                 "Phi ft-both+fleurs (VP)",
}

MODEL_ORDER = list(MODEL_STYLES.keys())


# load
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

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
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


_FLEURS_FT_KEYS   = {k for k in MODEL_STYLES if "fleurs_context" in k}
_EVAL_FT_KEYS     = {k for k in MODEL_STYLES if "/" in k and "fleurs" not in k}
_COMBINED_FT_KEYS = {k for k in MODEL_STYLES if k.endswith("_fleurs_mixed")}

_ZEROSHOT_KEYS = {"qwen_omni", "phi_multimodal"}

MODEL_SUBSETS = {
    "all":               None,
    "zeroshot_fleurs":   lambda k: k in _ZEROSHOT_KEYS or k in _FLEURS_FT_KEYS,
    "zeroshot_eval":     lambda k: k in _ZEROSHOT_KEYS or k in _EVAL_FT_KEYS,
    "zeroshot_combined": lambda k: k in _ZEROSHOT_KEYS or k in _COMBINED_FT_KEYS,
}

SUBSET_SUFFIX = {
    "all":               "",
    "zeroshot_fleurs":   "_fleurs_ft",
    "zeroshot_eval":     "_eval_ft",
    "zeroshot_combined": "_combined_ft",
}


def _filter_models(models, subset_fn):
    if subset_fn is None:
        return models
    return {k: v for k, v in models.items() if subset_fn(k)}


_CONTEXT_CONDITIONS    = ["no_context", "word_target", "sentence_target", "sentences_5_target", "sentences_10_target"]
_ATTACK_CONDITIONS     = ["no_context", "word_context", "sentence_context", "sentences_5_context", "sentences_10_context"]
_MITIGATION_CONDITIONS = ["no_context", None, "sentences_2_mixed", "sentences_5_mixed", "sentences_10_mixed"]
_MIXED_CONDITIONS      = ["no_context", "word_mixed", "sentences_2_mixed", "sentences_5_mixed", "sentences_10_mixed"]
_COND_LABELS           = ["no context", "word", "1/2-sent", "5-sent", "10-sent"]

_SHARED_STYLE = {
    "base":     {"color": "#90CAF9", "lw": 1.5, "marker": "o"},
    "ctx_ft":        {"color": "#1565C0", "lw": 1.5, "marker": "o"},
    "combined":      {"color": "#FF8F00", "lw": 1.5, "marker": "s"},
    "mitigation_ft": {"color": "#BF360C", "lw": 1.5, "marker": "s"},
}

def _make_baseline_lines(prefix=""):
    return {
        "qwen": [
            ("qwen_omni",                                            "Base model",                        _SHARED_STYLE["base"]),
            ("qwen/fleurs_context_mixed",                            "Prompt-adapted",                    _SHARED_STYLE["ctx_ft"]),
            (f"qwen/{prefix}target_word_fleurs_mixed",               "Acoustic word FT + prompt-adapted", _SHARED_STYLE["combined"]),
        ],
        "phi": [
            ("phi_multimodal",                                       "Base model",                        _SHARED_STYLE["base"]),
            ("phi/fleurs_context_mixed",                             "Prompt-adapted",                    _SHARED_STYLE["ctx_ft"]),
            (f"phi/{prefix}target_word_fleurs_mixed",                "Acoustic word FT + prompt-adapted", _SHARED_STYLE["combined"]),
        ],
    }

def _make_mitigation_lines(prefix=""):
    lines = _make_baseline_lines(prefix)
    for family in lines:
        lines[family].append(
            (f"{'qwen' if family == 'qwen' else 'phi'}/{prefix}both_fleurs_mixed",
             "Both words FT + prompt-adapted", _SHARED_STYLE["mitigation_ft"])
        )
    return lines

def _make_attack_lines(prefix=""):
    return {
        "qwen": [
            ("qwen_omni",                                            "Base model",                        _SHARED_STYLE["base"]),
            ("qwen/fleurs_context_mixed",                            "Prompt-adapted",                    _SHARED_STYLE["ctx_ft"]),
            (f"qwen/{prefix}context_word_fleurs_mixed",              "Context word FT + prompt-adapted",  _SHARED_STYLE["combined"]),
            (f"qwen/{prefix}both_fleurs_mixed",                      "Both words FT + prompt-adapted",    _SHARED_STYLE["mitigation_ft"]),
        ],
        "phi": [
            ("phi_multimodal",                                       "Base model",                        _SHARED_STYLE["base"]),
            ("phi/fleurs_context_mixed",                             "Prompt-adapted",                    _SHARED_STYLE["ctx_ft"]),
            (f"phi/{prefix}context_word_fleurs_mixed",               "Context word FT + prompt-adapted",  _SHARED_STYLE["combined"]),
            (f"phi/{prefix}both_fleurs_mixed",                       "Both words FT + prompt-adapted",    _SHARED_STYLE["mitigation_ft"]),
        ],
    }

_BASELINE_LINES = _make_baseline_lines()
_ATTACK_LINES   = _make_attack_lines()


def plot_two_panel(all_models, lines_dict, conditions, labels, metric, ylabel, out_path, fmt=lambda v: v * 100):
    """Generic two-panel plot (Qwen | Phi)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    x = list(range(len(conditions)))

    for ax, (family, lines) in zip(axes, lines_dict.items()):
        for key, label, style in lines:
            if key not in all_models:
                continue
            vals = [fmt(all_models[key].get(c, {}).get(metric, float("nan"))) for c in conditions]
            ax.plot(x, vals, color=style["color"], linewidth=style["lw"],
                    marker=style["marker"], markersize=5, label=label, linestyle="-")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
        ax.set_title("Qwen2.5-Omni-7B" if family == "qwen" else "Phi-4-Multimodal", fontsize=12)
        ax.axvline(x=0.5, color="#cccccc", linewidth=1.0, linestyle=":", zorder=0)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.tick_params(axis="y", labelsize=11)

    axes[0].set_ylabel(ylabel, fontsize=12)
    handles = [
        mlines.Line2D([], [], color=style["color"], linestyle="-",
                      marker=style["marker"], markersize=6,
                      linewidth=style["lw"], label=label)
        for _, label, style in list(lines_dict.values())[0]
    ]
    fig.legend(handles=handles, fontsize=11, loc="lower center",
               bbox_to_anchor=(0.5, -0.10), ncol=len(handles), frameon=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_single_panel(all_models, lines, conditions, labels, metric, ylabel, out_path, fmt=lambda v: v * 100):
    """Single-panel line plot (Qwen only)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    x = list(range(len(conditions)))
    for key, label, style in lines:
        if key not in all_models:
            continue
        vals = [fmt(all_models[key].get(c, {}).get(metric, float("nan"))) for c in conditions]
        ax.plot(x, vals, color=style["color"], linewidth=style["lw"],
                marker=style["marker"], markersize=5, label=label, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
    ax.set_title("Qwen2.5-Omni-7B", fontsize=12)
    ax.axvline(x=0.5, color="#cccccc", linewidth=1.0, linestyle=":", zorder=0)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    handles = [
        mlines.Line2D([], [], color=style["color"], linestyle="-",
                      marker=style["marker"], markersize=6,
                      linewidth=style["lw"], label=label)
        for _, label, style in lines
    ]
    ax.legend(handles=handles, fontsize=11, loc="best", frameon=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


_MIXED_CONDS_FULL = ["no_context", "word_mixed", "sentences_2_mixed", "sentences_5_mixed", "sentences_10_mixed"]


def plot_baseline_with_mitigation(all_models, lines_dict, metric, ylabel, out_path, fmt=lambda v: v * 100):
    """Baseline (solid, target-word context) + mitigation (dotted, both-word context) on same axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
    x = [0, 1, 2, 3, 4]

    for ax, (family, lines) in zip(axes, lines_dict.items()):
        for key, label, style in lines:
            if key not in all_models:
                continue
            base_vals = [fmt(all_models[key].get(c, {}).get(metric, float("nan")))
                         for c in _CONTEXT_CONDITIONS]
            ax.plot(x, base_vals, color=style["color"], linewidth=style["lw"],
                    marker=style["marker"], markersize=5, linestyle="-")
            mit_vals = [fmt(all_models[key].get(c, {}).get(metric, float("nan")))
                        for c in _MIXED_CONDS_FULL]
            ax.plot(x, mit_vals, color=style["color"], linewidth=style["lw"],
                    marker=style["marker"], markersize=6, linestyle=":",
                    markerfacecolor="white", markeredgewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(["no context", "word", "1-sent", "5-sent", "10-sent"],
                           rotation=0, ha="center", fontsize=11)
        ax.set_title("Qwen2.5-Omni-7B" if family == "qwen" else "Phi-4-Multimodal", fontsize=12)
        ax.axvline(x=0.5, color="#cccccc", linewidth=1.0, linestyle=":", zorder=0)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.tick_params(axis="y", labelsize=11)

    axes[0].set_ylabel(ylabel, fontsize=12)
    model_handles = [
        mlines.Line2D([], [], color=style["color"], linestyle="-",
                      marker=style["marker"], markersize=5,
                      linewidth=style["lw"], label=label)
        for _, label, style in list(lines_dict.values())[0]
    ]
    style_handles = [
        mlines.Line2D([], [], color="grey", linestyle=":", linewidth=1.5, marker="o", markersize=6,
                      markerfacecolor="white", markeredgewidth=1.5, label="prompt mitigated"),
    ]
    fig.legend(handles=model_handles + style_handles, fontsize=11, loc="lower center",
               bbox_to_anchor=(0.5, -0.10), ncol=len(model_handles) + 2, frameon=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def _attack_mit_panel(ax, all_models, lines, family, metric, ylabel, fmt):
    x_atk = [0, 1, 2, 3, 4]
    x_mit = [0, 2, 3, 4]
    mit_conds = ["no_context", "sentences_2_mixed", "sentences_5_mixed", "sentences_10_mixed"]
    for key, label, style in lines:
        if key not in all_models:
            continue
        atk_vals = [fmt(all_models[key].get(c, {}).get(metric, float("nan"))) for c in _ATTACK_CONDITIONS]
        ax.plot(x_atk, atk_vals, color=style["color"], linewidth=style["lw"],
                marker=style["marker"], markersize=5, linestyle="-")
        mit_vals = [fmt(all_models[key].get(c, {}).get(metric, float("nan"))) for c in mit_conds]
        ax.plot(x_mit, mit_vals, color=style["color"], linewidth=style["lw"],
                marker=style["marker"], markersize=6, linestyle=":",
                markerfacecolor="white", markeredgewidth=1.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["no context", "word", "1-sent", "5-sent", "10-sent"],
                       rotation=0, ha="center", fontsize=11)
    ax.set_title("Qwen2.5-Omni-7B" if family == "qwen" else "Phi-4-Multimodal", fontsize=12)
    ax.axvline(x=0.5, color="#cccccc", linewidth=1.0, linestyle=":", zorder=0)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.tick_params(axis="y", labelsize=11)


def _attack_mit_legend(fig, lines, n_extra_cols=2):
    model_handles = [
        mlines.Line2D([], [], color=style["color"], linestyle="-",
                      marker=style["marker"], markersize=5,
                      linewidth=style["lw"], label=label)
        for _, label, style in lines
    ]
    style_handles = [
        mlines.Line2D([], [], color="grey", linestyle=":", linewidth=1.5, marker="o", markersize=6,
                      markerfacecolor="white", markeredgewidth=1.5, label="prompt mitigated"),
    ]
    fig.legend(handles=model_handles + style_handles, fontsize=11, loc="lower center",
               bbox_to_anchor=(0.5, -0.10), ncol=len(model_handles) + n_extra_cols, frameon=True)


def plot_attack_with_mitigation(all_models, lines_dict, metric, ylabel, out_path, fmt=lambda v: v * 100):
    """Attack (solid) + mitigation (dotted) on the same axes, two panels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
    for ax, (family, lines) in zip(axes, lines_dict.items()):
        _attack_mit_panel(ax, all_models, lines, family, metric, ylabel, fmt)
    axes[0].set_ylabel(ylabel, fontsize=12)
    _attack_mit_legend(fig, list(lines_dict.values())[0])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_attack_with_mitigation_single(all_models, lines_dict, family, metric, ylabel, out_path, fmt=lambda v: v * 100):
    """Attack (solid) + mitigation (dotted), single panel for one model family."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    _attack_mit_panel(ax, all_models, lines_dict[family], family, metric, ylabel, fmt)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Qwen2.5-Omni-7B", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    lines = lines_dict[family]
    model_handles = [
        mlines.Line2D([], [], color=style["color"], linestyle="-",
                      marker=style["marker"], markersize=6,
                      linewidth=style["lw"],
                      label=label.replace("prompt-adapted", "prompt-adapt.").replace("Prompt-adapted", "Prompt-adapt."))
        for _, label, style in lines
    ]
    style_handles = [
        mlines.Line2D([], [], color="grey", linestyle=":", linewidth=1.5, marker="o", markersize=6,
                      markerfacecolor="white", markeredgewidth=1.5, label="prompt mitigated"),
    ]
    all_handles = model_handles + style_handles
    ncol = (len(all_handles) + 2) // 3
    fig.legend(handles=all_handles, fontsize=11, loc="lower center",
               bbox_to_anchor=(0.5, -0.25), ncol=ncol, frameon=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def _plot_all_sections(all_models, out_dir, baseline_lines, attack_lines, mitigation_lines=None, scatter_lines=None):
    """Generate plots for sections A, B and C."""
    plot_two_panel(
        all_models, baseline_lines, _CONTEXT_CONDITIONS, _COND_LABELS,
        metric="target_correct", ylabel="Acoustic word accuracy (%)",
        out_path=os.path.join(out_dir, "plot_results_a", "results_baseline_acoustic_accuracy.pdf"),
    )
    plot_two_panel(
        all_models, baseline_lines, _CONTEXT_CONDITIONS, _COND_LABELS,
        metric="background_wer", ylabel="Background WER (%)", fmt=lambda v: v * 100,
        out_path=os.path.join(out_dir, "plot_results_a", "results_baseline_wer.pdf"),
    )
    # Qwen-only versions for the paper body
    plot_single_panel(
        all_models, baseline_lines["qwen"], _CONTEXT_CONDITIONS, _COND_LABELS,
        metric="target_correct", ylabel="Acoustic word accuracy (%)",
        out_path=os.path.join(out_dir, "plot_results_a", "results_baseline_acoustic_accuracy_qwen.pdf"),
    )
    plot_single_panel(
        all_models, baseline_lines["qwen"], _CONTEXT_CONDITIONS, _COND_LABELS,
        metric="background_wer", ylabel="Background WER (%)", fmt=lambda v: v * 100,
        out_path=os.path.join(out_dir, "plot_results_a", "results_baseline_wer_qwen.pdf"),
    )
    plot_two_panel(
        all_models, attack_lines, _ATTACK_CONDITIONS, _COND_LABELS,
        metric="target_to_context", ylabel="Leakage rate (%)",
        out_path=os.path.join(out_dir, "plot_results_b", "results_attack_leakage.pdf"),
    )
    plot_two_panel(
        all_models, attack_lines, _ATTACK_CONDITIONS, _COND_LABELS,
        metric="background_wer", ylabel="Background WER (%)", fmt=lambda v: v * 100,
        out_path=os.path.join(out_dir, "plot_results_b", "results_attack_wer.pdf"),
    )
    plot_attack_with_mitigation(
        all_models, attack_lines,
        metric="target_to_context", ylabel="Leakage rate (%)",
        out_path=os.path.join(out_dir, "plot_results_c", "results_mitigation_leakage.pdf"),
    )
    plot_attack_with_mitigation_single(
        all_models, attack_lines, "qwen",
        metric="target_to_context", ylabel="Leakage rate (%)",
        out_path=os.path.join(out_dir, "plot_results_c", "results_mitigation_leakage_qwen.pdf"),
    )
    plot_baseline_with_mitigation(
        all_models, mitigation_lines or baseline_lines,
        metric="target_correct", ylabel="Acoustic word accuracy (%)",
        out_path=os.path.join(out_dir, "plot_results_a", "results_mitigation_acoustic_accuracy.pdf"),
    )
    if scatter_lines:
        plot_accuracy_vs_leakage(
            all_models, scatter_lines,
            out_path=os.path.join(out_dir, "general_plots", "results_accuracy_vs_leakage.pdf"),
        )
        for family in ("qwen", "phi"):
            plot_accuracy_vs_leakage_single(
                all_models, scatter_lines, family,
                out_path=os.path.join(out_dir, "general_plots", f"results_accuracy_vs_leakage_{family}.pdf"),
            )


_SCATTER_LEAK_CONDS  = ["no_context", "word_context",  "sentence_context"]
_SCATTER_ACC_CONDS   = ["no_context", "word_target",   "sentence_target"]
_SCATTER_COND_LABELS = ["no context", "word", "1-sent"]
_SCATTER_MARKERS     = ["o", "s", "^"]
_SCATTER_SIZES       = [120, 120, 120]


def _make_scatter_methods(prefix=""):
    """Methods for the accuracy-vs-leakage scatter plot.
    Each entry: (label, leakage_model_key, accuracy_model_key, style)
    X is taken from leakage_model under attack conditions,
    Y is taken from accuracy_model under helpful-context conditions.
    """
    return {
        "qwen": [
            ("Base model",                       "qwen_omni",                              "qwen_omni",                             _SHARED_STYLE["base"]),
            ("Prompt-adapted",                   "qwen/fleurs_context_mixed",              "qwen/fleurs_context_mixed",             _SHARED_STYLE["ctx_ft"]),
            ("Context word FT + prompt-adapted", f"qwen/{prefix}context_word_fleurs_mixed", f"qwen/{prefix}target_word_fleurs_mixed", _SHARED_STYLE["combined"]),
            ("Both words FT + prompt-adapted",   f"qwen/{prefix}both_fleurs_mixed",         f"qwen/{prefix}both_fleurs_mixed",        _SHARED_STYLE["mitigation_ft"]),
        ],
        "phi": [
            ("Base model",                       "phi_multimodal",                         "phi_multimodal",                        _SHARED_STYLE["base"]),
            ("Prompt-adapted",                   "phi/fleurs_context_mixed",               "phi/fleurs_context_mixed",              _SHARED_STYLE["ctx_ft"]),
            ("Context word FT + prompt-adapted", f"phi/{prefix}context_word_fleurs_mixed",  f"phi/{prefix}target_word_fleurs_mixed",  _SHARED_STYLE["combined"]),
            ("Both words FT + prompt-adapted",   f"phi/{prefix}both_fleurs_mixed",          f"phi/{prefix}both_fleurs_mixed",         _SHARED_STYLE["mitigation_ft"]),
        ],
    }


_SCATTER_MIT_CONDS   = ["word_mixed", "sentences_2_mixed"]
_SCATTER_MIT_INDICES = [1, 2]


def _scatter_panel(ax, all_models, methods, family):
    for label, leak_key, acc_key, style in methods:
        if leak_key not in all_models or acc_key not in all_models:
            continue
        leak_model = all_models[leak_key]
        acc_model  = all_models[acc_key]
        color      = style["color"]
        for i, (lc, ac) in enumerate(zip(_SCATTER_LEAK_CONDS, _SCATTER_ACC_CONDS)):
            x = leak_model.get(lc, {}).get("target_to_context", float("nan")) * 100
            y = acc_model.get(ac,  {}).get("target_correct",    float("nan")) * 100
            ax.scatter(x, y, color=color, s=_SCATTER_SIZES[i],
                       marker=_SCATTER_MARKERS[i], zorder=3, edgecolors="white", linewidths=0.5)
        for i, mc in zip(_SCATTER_MIT_INDICES, _SCATTER_MIT_CONDS):
            x = leak_model.get(mc, {}).get("target_to_context", float("nan")) * 100
            y = leak_model.get(mc, {}).get("target_correct",    float("nan")) * 100
            ax.scatter(x, y, s=_SCATTER_SIZES[i], marker=_SCATTER_MARKERS[i], zorder=3,
                       facecolors="white", edgecolors=color, linewidths=1.5)
    ax.plot(0.0, 1.0, marker="*", markersize=14, color="dimgray",
            transform=ax.transAxes, zorder=5, linestyle="None",
            clip_on=False)
    ax.text(0.0, 1.06, "ideal", transform=ax.transAxes,
            fontsize=10, color="dimgray", va="bottom", ha="left", clip_on=False)
    ax.set_xlabel("Leakage rate (%)", fontsize=12)
    ax.set_title("Qwen2.5-Omni-7B" if family == "qwen" else "Phi-4-Multimodal", fontsize=12)
    ax.grid(linewidth=0.4, alpha=0.5)


def _scatter_legend(fig, methods):
    model_handles = [
        mlines.Line2D([], [], color=style["color"], linestyle="None", marker="o", markersize=7,
                      label=label)
        for label, _, _, style in methods
    ]
    cond_handles = [
        mlines.Line2D([], [], color="grey", linestyle="None",
                      marker=_SCATTER_MARKERS[i], markersize=np.sqrt(_SCATTER_SIZES[i]),
                      label=_SCATTER_COND_LABELS[i])
        for i in range(len(_SCATTER_COND_LABELS))
    ]
    style_handles = [
        mlines.Line2D([], [], color="grey", linestyle="None", marker="o", markersize=7,
                      markerfacecolor="white", markeredgewidth=1.5, label="prompt mitigated"),
    ]
    n_cols = max(len(model_handles), len(cond_handles) + len(style_handles))
    fig.legend(handles=model_handles + cond_handles + style_handles,
               fontsize=10, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               ncol=n_cols, frameon=True)


def plot_accuracy_vs_leakage(all_models, methods_dict, out_path):
    """Two-panel scatter (Qwen | Phi)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (family, methods) in zip(axes, methods_dict.items()):
        _scatter_panel(ax, all_models, methods, family)
    axes[0].set_ylabel("Acoustic word accuracy (%)", fontsize=12)
    _scatter_legend(fig, list(methods_dict.values())[0])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_accuracy_vs_leakage_single(all_models, methods_dict, family, out_path):
    """Single-panel scatter for one model family."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    _scatter_panel(ax, all_models, methods_dict[family], family)
    ax.set_ylabel("Acoustic word accuracy (%)", fontsize=12)

    # Model colours legend — below the plot
    methods = methods_dict[family]
    model_handles = [
        mlines.Line2D([], [], color=style["color"], linestyle="None", marker="o", markersize=7,
                      label=label)
        for label, _, _, style in methods
    ]
    fig.legend(handles=model_handles, fontsize=10, loc="lower center",
               bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True)

    # Condition markers + prompt mitigated — inside bottom right corner
    cond_handles = [
        mlines.Line2D([], [], color="grey", linestyle="None",
                      marker=_SCATTER_MARKERS[i], markersize=np.sqrt(_SCATTER_SIZES[i]),
                      label=_SCATTER_COND_LABELS[i])
        for i in range(len(_SCATTER_COND_LABELS))
    ]
    style_handles = [
        mlines.Line2D([], [], color="grey", linestyle="None", marker="o", markersize=7,
                      markerfacecolor="white", markeredgewidth=1.5, label="prompt mitigated"),
    ]
    ax.legend(handles=cond_handles + style_handles, fontsize=10,
              loc="lower right", frameon=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def make_plots(eval_root="generated_eval", out_dir="generated_eval", dataset_prefix=""):
    all_models       = load_results(eval_root)
    baseline_lines   = _make_baseline_lines(dataset_prefix)
    attack_lines     = _make_attack_lines(dataset_prefix)
    mitigation_lines = _make_mitigation_lines(dataset_prefix)
    scatter_lines    = _make_scatter_methods(dataset_prefix)
    _plot_all_sections(all_models, out_dir, baseline_lines, attack_lines, mitigation_lines, scatter_lines)


_SIM_UNSPLIT_CONDS  = {0, 1}  # no_context and word_context: sentence similarity not applicable
_DIST_UNSPLIT_CONDS = {0}     # no_context only: phoneme distance is meaningful for word_context


def _plot_sim_bars(ax, get_vals, color, x, width, groups, hatches, unsplit_conds):
    """Single averaged bar for unsplit conditions, grouped bars for the rest."""
    for i_cond in range(len(_SIM_ATTACK_CONDS)):
        if i_cond in unsplit_conds:
            vals_per_group = [get_vals(g) for g in groups]
            avg = float(np.nanmean([v[i_cond] for v in vals_per_group]))
            ax.bar(x[i_cond], avg, width * len(groups),
                   color=color, edgecolor="white", linewidth=0.5, alpha=0.9)
        else:
            for i_group, group in enumerate(groups):
                vals = get_vals(group)
                ax.bar(x[i_cond] + (i_group - 1) * width, vals[i_cond], width,
                       color=color, hatch=hatches[group],
                       edgecolor="white", linewidth=0.5, alpha=0.9)


_SIM_GROUPS       = ["distinct", "related", "near-identical"]
_SIM_HATCHES      = {"distinct": "", "related": "///", "near-identical": "xxx"}
_SIM_GROUP_LABELS = {"distinct": "Distinct (≤0.4)", "related": "Related (0.4–0.7)", "near-identical": "Near-identical (>0.7)"}
# map from group name used in filenames to new display names
_SIM_GROUP_FILE_MAP = {"different": "distinct", "similar": "related", "near-copy": "near-identical"}
_SIM_ATTACK_CONDS = ["no_context", "word_context", "sentence_context", "sentences_5_context", "sentences_10_context"]
_SIM_COND_LABELS  = ["no ctx", "word", "1-sent", "5-sent", "10-sent"]

_SIM_MODEL_COLORS = {
    "Prompt-adapted":                    _SHARED_STYLE["ctx_ft"]["color"],
    "Context word FT + prompt-adapted":  _SHARED_STYLE["combined"]["color"],
}

_SIM_MODELS = {
    "qwen": [
        ("fleurs/qwen/fleurs_context_mixed",                        "Prompt-adapted"),
        ("acl6060/qwen/fleurs_context_mixed",                       "Prompt-adapted"),
        ("voxpopuli/qwen/fleurs_context_mixed",                     "Prompt-adapted"),
        ("fleurs/qwen/context_word_fleurs_mixed",                   "Context word FT + prompt-adapted"),
        ("acl6060/qwen/acl6060_context_word_fleurs_mixed",          "Context word FT + prompt-adapted"),
        ("voxpopuli/qwen/voxpopuli_context_word_fleurs_mixed",      "Context word FT + prompt-adapted"),
    ],
    "phi": [
        ("fleurs/phi/fleurs_context_mixed",                         "Prompt-adapted"),
        ("acl6060/phi/fleurs_context_mixed",                        "Prompt-adapted"),
        ("voxpopuli/phi/fleurs_context_mixed",                      "Prompt-adapted"),
        ("fleurs/phi/context_word_fleurs_mixed",                    "Context word FT + prompt-adapted"),
        ("acl6060/phi/acl6060_context_word_fleurs_mixed",           "Context word FT + prompt-adapted"),
        ("voxpopuli/phi/voxpopuli_context_word_fleurs_mixed",       "Context word FT + prompt-adapted"),
    ],
}


def plot_similarity_analysis(sim_root: str, out_dir: str, metric: str = "target_to_context", ylabel: str = "Leakage rate (%)"):
    """Bar chart: model colors match main plots, hatch patterns encode similarity bin."""
    from matplotlib.patches import Patch

    raw = {}
    for family, models in _SIM_MODELS.items():
        for model_key, model_label in models:
            for file_name, group in _SIM_GROUP_FILE_MAP.items():
                path = os.path.join(sim_root, model_key.replace("/", os.sep), f"{file_name}.json")
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    conds = json.load(f)["conditions"]
                vals = [conds.get(c, {}).get(metric, float("nan")) * 100
                        for c in _SIM_ATTACK_CONDS]
                raw.setdefault((family, model_label, group), []).append(vals)
    data = {k: list(np.nanmean(v, axis=0)) for k, v in raw.items()}

    if not data:
        print(f"No similarity analysis results found in {sim_root}")
        return

    unique_labels = list(dict.fromkeys(lbl for _, lbl in _SIM_MODELS["qwen"]))
    n_models = len(unique_labels)
    fig, axes = plt.subplots(2, n_models, figsize=(12, 6), sharey=True)
    x = np.arange(len(_SIM_COND_LABELS))
    width = 0.28

    for row, family in enumerate(["qwen", "phi"]):
        for col, model_label in enumerate(unique_labels):
            ax = axes[row][col]
            color = _SIM_MODEL_COLORS[model_label]
            _plot_sim_bars(ax,
                           lambda g, f=family, m=model_label: data.get((f, m, g), [float("nan")] * len(_SIM_ATTACK_CONDS)),
                           color, x, width, _SIM_GROUPS, _SIM_HATCHES, _SIM_UNSPLIT_CONDS)
            ax.set_xticks(x)
            ax.set_xticklabels(_SIM_COND_LABELS, fontsize=11)
            ax.set_title(model_label, fontsize=12)
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
            ax.tick_params(axis="y", labelsize=11)
            if col == 0:
                family_name = "Qwen2.5-Omni" if family == "qwen" else "Phi-4-Multimodal"
                ax.set_ylabel(f"{family_name}\n{ylabel}", fontsize=12)

    model_handles = [Patch(facecolor=_SIM_MODEL_COLORS[lbl], label=lbl)
                     for lbl in _SIM_MODEL_COLORS]
    bin_handles   = [Patch(facecolor="grey", hatch=_SIM_HATCHES[g], edgecolor="white",
                           label=_SIM_GROUP_LABELS[g]) for g in _SIM_GROUPS]
    fig.legend(handles=model_handles + [Patch(visible=False)] + bin_handles,
               fontsize=11, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=len(model_handles) + 1 + len(bin_handles),
               frameon=True)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "similarity_analysis_leakage.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_similarity_analysis_qwen(sim_root: str, out_dir: str, metric: str = "target_to_context", ylabel: str = "Leakage rate (%)"):
    """Single-panel bar chart for Qwen context_word_fleurs_mixed only."""
    from matplotlib.patches import Patch

    model_label = "Context word FT + prompt-adapted"
    color       = _SIM_MODEL_COLORS[model_label]
    model_keys  = [k for k, lbl in _SIM_MODELS["qwen"] if lbl == model_label]

    raw = {}
    for model_key in model_keys:
        for file_name, group in _SIM_GROUP_FILE_MAP.items():
            path = os.path.join(sim_root, model_key.replace("/", os.sep), f"{file_name}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                conds = json.load(f)["conditions"]
            vals = [conds.get(c, {}).get(metric, float("nan")) * 100 for c in _SIM_ATTACK_CONDS]
            raw.setdefault(group, []).append(vals)
    data = {g: list(np.nanmean(v, axis=0)) for g, v in raw.items()}

    if not data:
        print(f"No similarity analysis results found in {sim_root}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    x     = np.arange(len(_SIM_COND_LABELS))
    width = 0.28
    _plot_sim_bars(ax, lambda g: data.get(g, [float("nan")] * len(_SIM_ATTACK_CONDS)),
                   color, x, width, _SIM_GROUPS, _SIM_HATCHES, _SIM_UNSPLIT_CONDS)
    ax.set_xticks(x)
    ax.set_xticklabels(_SIM_COND_LABELS, fontsize=11)
    ax.set_title("Qwen2.5-Omni-7B (Context word FT + prompt-adapted)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.tick_params(axis="y", labelsize=11)
    bin_handles = [Patch(facecolor=color, hatch=_SIM_HATCHES[g], edgecolor="white",
                         label=_SIM_GROUP_LABELS[g]) for g in _SIM_GROUPS]
    ax.legend(handles=bin_handles, fontsize=11, loc="best", frameon=True)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "similarity_analysis_leakage_qwen.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


_DIST_GROUPS  = [1, 2]
_DIST_COLORS  = {1: _SHARED_STYLE["ctx_ft"]["color"], 2: _SHARED_STYLE["combined"]["color"]}
_DIST_LABELS  = {1: "Distance 1", 2: "Distance 2"}
_DIST_HATCHES = {1: "", 2: "///"}

_DIST_MODELS = {
    "qwen": [
        ("fleurs/qwen/fleurs_context_mixed",                        "Prompt-adapted"),
        ("acl6060/qwen/fleurs_context_mixed",                       "Prompt-adapted"),
        ("voxpopuli/qwen/fleurs_context_mixed",                     "Prompt-adapted"),
        ("fleurs/qwen/context_word_fleurs_mixed",                   "Context word FT + prompt-adapted"),
        ("acl6060/qwen/acl6060_context_word_fleurs_mixed",          "Context word FT + prompt-adapted"),
        ("voxpopuli/qwen/voxpopuli_context_word_fleurs_mixed",      "Context word FT + prompt-adapted"),
    ],
    "phi": [
        ("fleurs/phi/fleurs_context_mixed",                         "Prompt-adapted"),
        ("acl6060/phi/fleurs_context_mixed",                        "Prompt-adapted"),
        ("voxpopuli/phi/fleurs_context_mixed",                      "Prompt-adapted"),
        ("fleurs/phi/context_word_fleurs_mixed",                    "Context word FT + prompt-adapted"),
        ("acl6060/phi/acl6060_context_word_fleurs_mixed",           "Context word FT + prompt-adapted"),
        ("voxpopuli/phi/voxpopuli_context_word_fleurs_mixed",       "Context word FT + prompt-adapted"),
    ],
}


def plot_distance_analysis(dist_root: str, out_dir: str, metric: str = "target_to_context", ylabel: str = "Leakage rate (%)"):
    """Bar chart comparing metric for phoneme distance 1 vs 2."""
    from matplotlib.patches import Patch

    raw = {}
    for family, models in _DIST_MODELS.items():
        for model_key, model_label in models:
            for dist in _DIST_GROUPS:
                path = os.path.join(dist_root, model_key.replace("/", os.sep), f"distance_{dist}.json")
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    conds = json.load(f)["conditions"]
                vals = [conds.get(c, {}).get(metric, float("nan")) * 100
                        for c in _SIM_ATTACK_CONDS]
                raw.setdefault((family, model_label, dist), []).append(vals)
    data = {k: list(np.nanmean(v, axis=0)) for k, v in raw.items()}

    if not data:
        print(f"No distance analysis results found in {dist_root}")
        return

    unique_labels = list(dict.fromkeys(lbl for _, lbl in _DIST_MODELS["qwen"]))
    n_models = len(unique_labels)
    fig, axes = plt.subplots(2, n_models, figsize=(12, 6), sharey=True)
    x = np.arange(len(_SIM_COND_LABELS))
    width = 0.35

    for row, family in enumerate(["qwen", "phi"]):
        for col, model_label in enumerate(unique_labels):
            ax = axes[row][col]
            color = _SIM_MODEL_COLORS[model_label]
            _plot_sim_bars(ax,
                           lambda d, f=family, m=model_label: data.get((f, m, d), [float("nan")] * len(_SIM_ATTACK_CONDS)),
                           color, x, width, _DIST_GROUPS, _DIST_HATCHES, _DIST_UNSPLIT_CONDS)
            ax.set_xticks(x)
            ax.set_xticklabels(_SIM_COND_LABELS, fontsize=11)
            ax.set_title(model_label, fontsize=12)
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
            ax.tick_params(axis="y", labelsize=11)
            if col == 0:
                family_name = "Qwen2.5-Omni" if family == "qwen" else "Phi-4-Multimodal"
                ax.set_ylabel(f"{family_name}\n{ylabel}", fontsize=12)

    model_handles = [Patch(facecolor=_SIM_MODEL_COLORS[lbl], label=lbl)
                     for lbl in _SIM_MODEL_COLORS]
    dist_handles  = [Patch(facecolor="grey", hatch=_DIST_HATCHES[d], edgecolor="white",
                           label=_DIST_LABELS[d]) for d in _DIST_GROUPS]
    fig.legend(handles=model_handles + [Patch(visible=False)] + dist_handles,
               fontsize=11, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=len(model_handles) + 1 + len(dist_handles),
               frameon=True)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "distance_analysis_leakage.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_distance_analysis_qwen(dist_root: str, out_dir: str, metric: str = "target_to_context", ylabel: str = "Leakage rate (%)"):
    """Single-panel bar chart for Qwen context_word_fleurs_mixed only."""
    from matplotlib.patches import Patch

    model_label = "Context word FT + prompt-adapted"
    color       = _SIM_MODEL_COLORS[model_label]
    model_keys  = [k for k, lbl in _DIST_MODELS["qwen"] if lbl == model_label]

    raw = {}
    for model_key in model_keys:
        for dist in _DIST_GROUPS:
            path = os.path.join(dist_root, model_key.replace("/", os.sep), f"distance_{dist}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                conds = json.load(f)["conditions"]
            vals = [conds.get(c, {}).get(metric, float("nan")) * 100 for c in _SIM_ATTACK_CONDS]
            raw.setdefault(dist, []).append(vals)
    data = {d: list(np.nanmean(v, axis=0)) for d, v in raw.items()}

    if not data:
        print(f"No distance analysis results found in {dist_root}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    x     = np.arange(len(_SIM_COND_LABELS))
    width = 0.35
    _plot_sim_bars(ax, lambda d: data.get(d, [float("nan")] * len(_SIM_ATTACK_CONDS)),
                   color, x, width, _DIST_GROUPS, _DIST_HATCHES, _DIST_UNSPLIT_CONDS)
    ax.set_xticks(x)
    ax.set_xticklabels(_SIM_COND_LABELS, fontsize=11)
    ax.set_title("Qwen2.5-Omni-7B (Context word FT + prompt-adapted)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.tick_params(axis="y", labelsize=11)
    dist_handles = [Patch(facecolor=color, hatch=_DIST_HATCHES[d], edgecolor="white",
                          label=_DIST_LABELS[d]) for d in _DIST_GROUPS]
    ax.legend(handles=dist_handles, fontsize=11, loc="best", frameon=True)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "distance_analysis_leakage_qwen.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


_MIT_CONDS = ["no_context", "sentences_2_mixed", "sentences_5_mixed", "sentences_10_mixed"]
_MIT_X     = [0, 2, 3, 4]
_ATK_X     = [0, 1, 2, 3, 4]

_DATASET_LABELS = {"fleurs": "FLEURS", "acl6060": "ACL 6060", "voxpopuli": "VoxPopuli"}


def plot_attack_all_datasets(eval_root: str, datasets: list[str], out_dir: str,
                             metric: str = "target_to_context", ylabel: str = "Leakage rate (%)"):
    """3-row × 2-col appendix figure: one row per dataset, columns = Qwen / Phi.
    Attack shown solid, mitigation dotted (matching plot_attack_with_mitigation style)."""
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 3.5 * len(datasets)), sharey=False)

    for row, dataset in enumerate(datasets):
        dataset_root = os.path.join(eval_root, dataset)
        if not os.path.isdir(dataset_root):
            continue
        all_models   = load_results(dataset_root)
        prefix       = "" if dataset == "fleurs" else f"{dataset}_"
        attack_lines = _make_attack_lines(prefix)

        for col, (family, lines) in enumerate(attack_lines.items()):
            ax = axes[row][col]
            for key, label, style in lines:
                if key not in all_models:
                    continue
                atk_vals = [all_models[key].get(c, {}).get(metric, float("nan")) * 100
                            for c in _ATTACK_CONDITIONS]
                ax.plot(_ATK_X, atk_vals, color=style["color"], linewidth=style["lw"],
                        marker=style["marker"], markersize=5, linestyle="-")
                mit_vals = [all_models[key].get(c, {}).get(metric, float("nan")) * 100
                            for c in _MIT_CONDS]
                ax.plot(_MIT_X, mit_vals, color=style["color"], linewidth=style["lw"],
                        marker=style["marker"], markersize=6, linestyle=":",
                        markerfacecolor="white", markeredgewidth=1.5)
            ax.set_xticks(_ATK_X)
            ax.set_xticklabels(["no context", "word", "1-sent", "5-sent", "10-sent"],
                               rotation=0, ha="center", fontsize=11)
            model_name = "Qwen2.5-Omni-7B" if family == "qwen" else "Phi-4-Multimodal"
            ax.set_title(f"{_DATASET_LABELS.get(dataset, dataset)} — {model_name}", fontsize=12)
            ax.axvline(x=0.5, color="#cccccc", linewidth=1.0, linestyle=":", zorder=0)
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
            ax.tick_params(axis="y", labelsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=12)

    model_handles = [
        mlines.Line2D([], [], color=style["color"], linestyle="-",
                      marker=style["marker"], markersize=5, linewidth=style["lw"], label=label)
        for _, label, style in list(_make_attack_lines("").values())[0]
    ]
    style_handles = [
        mlines.Line2D([], [], color="grey", linestyle=":", linewidth=1.5, marker="o", markersize=6,
                      markerfacecolor="white", markeredgewidth=1.5, label="prompt mitigated"),
    ]
    fig.legend(handles=model_handles + style_handles, fontsize=11, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=len(model_handles) + 1, frameon=True)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results_attack_leakage_all_datasets.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def load_results_averaged(eval_root: str, datasets: list[str]) -> dict:
    """Load results from multiple dataset subfolders and average metrics across them."""
    import numpy as np

    all_dataset_results = []
    for dataset in datasets:
        dataset_root = os.path.join(eval_root, dataset)
        if not os.path.isdir(dataset_root):
            print(f"Warning: {dataset_root} not found, skipping.")
            continue
        results = load_results(dataset_root)
        if results:
            all_dataset_results.append(results)

    if not all_dataset_results:
        return {}

    # Collect all model keys present in any dataset
    all_keys = set()
    for r in all_dataset_results:
        all_keys.update(r.keys())

    averaged = {}
    for key in MODEL_ORDER:
        if key not in all_keys:
            continue
        # Collect conditions dicts from each dataset that has this model
        per_dataset = [r[key] for r in all_dataset_results if key in r]
        if not per_dataset:
            continue
        # Average each metric across datasets
        all_conditions = set()
        for d in per_dataset:
            all_conditions.update(d.keys())
        avg_conds = {}
        for cond in all_conditions:
            all_metrics = set()
            for d in per_dataset:
                if cond in d:
                    all_metrics.update(d[cond].keys())
            avg_conds[cond] = {}
            for metric in all_metrics:
                vals = [d[cond][metric] for d in per_dataset if cond in d and metric in d[cond]]
                avg_conds[cond][metric] = float(np.mean(vals)) if vals else float("nan")
        averaged[key] = avg_conds

    return averaged


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", default="generated_eval")
    parser.add_argument("--out_dir",   default="generated_eval")
    parser.add_argument("--average_datasets", nargs="+", default=None,
                        help="Average results across these dataset subfolders.")
    parser.add_argument("--similarity_analysis", default=None,
                        help="If set, run similarity group analysis plot from this root folder.")
    parser.add_argument("--distance_analysis", default=None,
                        help="If set, run phoneme distance analysis plot from this root folder.")
    parser.add_argument("--dataset_prefix", default="",
                        help="Prefix for dataset-specific combined model keys, e.g. 'acl6060_' or 'voxpopuli_'.")
    args = parser.parse_args()

    dataset_prefix   = args.dataset_prefix or ""
    baseline_lines   = _make_baseline_lines(dataset_prefix)
    attack_lines     = _make_attack_lines(dataset_prefix)
    mitigation_lines = _make_mitigation_lines(dataset_prefix)
    scatter_lines    = _make_scatter_methods(dataset_prefix)

    if args.average_datasets:
        all_models = load_results_averaged(args.eval_root, args.average_datasets)
        os.makedirs(args.out_dir, exist_ok=True)
        _plot_all_sections(all_models, args.out_dir, baseline_lines, attack_lines, mitigation_lines, scatter_lines)
        plot_attack_all_datasets(args.eval_root, args.average_datasets, args.out_dir)
    elif args.similarity_analysis:
        plot_similarity_analysis(args.similarity_analysis, args.out_dir)
        plot_similarity_analysis_qwen(args.similarity_analysis, args.out_dir)
    elif args.distance_analysis:
        plot_distance_analysis(args.distance_analysis, args.out_dir)
        plot_distance_analysis_qwen(args.distance_analysis, args.out_dir)
    else:
        make_plots(args.eval_root, args.out_dir, dataset_prefix=dataset_prefix)
