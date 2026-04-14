import argparse
import json
import os
import re

from jiwer import cer, wer, process_words, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ExpandCommonEnglishContractions

normalizer = Compose([
    ToLowerCase(),
    ExpandCommonEnglishContractions(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])

CONDITIONS = [
    "no_context",
    "word_context",
    "word_target",
    "sentence_context",
    "sentence_target",
    "sentences_2_mixed",
    "sentences_5_context",
    "sentences_5_target",
    "sentences_5_mixed",
    "sentences_10_context",
    "sentences_10_target",
    "sentences_10_mixed",
]

_BG_PLACEHOLDER = "targetword"


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def contains_word(text, word):
    return bool(re.search(r'\b' + re.escape(word.lower()) + r'\b', normalizer(text)))


def _mask_words(text: str, words: list[str]) -> str:
    """Replace whole-word occurrences of each word with _BG_PLACEHOLDER (word-boundary safe)."""
    for word in words:
        text = re.sub(r'\b' + re.escape(word.lower()) + r'\b', _BG_PLACEHOLDER, text)
    return text


def word_match_rates(samples, condition):
    target_hits, context_hits, neither_hits = 0, 0, 0
    for s in samples:
        hyp = s["predicted"][condition]
        has_target = contains_word(hyp, s["target_word"])
        has_context = contains_word(hyp, s["context_word"])
        if has_target:
            target_hits += 1
        if has_context:
            context_hits += 1
        if not has_target and not has_context:
            neither_hits += 1
    n = len(samples)
    return target_hits / n, context_hits / n, neither_hits / n


def compute_background_wer(samples, condition):
    """WER with target_word masked in reference, and both target_word and context_word
    masked in hypothesis. This isolates whether context hurts transcription quality
    beyond the target/context word confusion itself."""
    masked_refs, masked_hyps = [], []
    for s in samples:
        ref = _mask_words(normalizer(s["reference"]), [s["target_word"]])
        hyp = _mask_words(normalizer(s["predicted"][condition]), [s["target_word"], s["context_word"]])
        masked_refs.append(ref)
        masked_hyps.append(hyp)
    return wer(masked_refs, masked_hyps)


def compute_target_alignment_stats(samples, condition):
    """Use word-level alignment to classify what happened at the target word position:
    correctly predicted, substituted with context_word, substituted with other, or deleted."""
    correct, to_context, to_other, deleted = 0, 0, 0, 0
    total = 0

    for s in samples:
        ref_norm = normalizer(s["reference"])
        hyp_norm = normalizer(s["predicted"][condition])
        target = s["target_word"].lower()
        context = s["context_word"].lower()

        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()

        target_indices = [i for i, w in enumerate(ref_words) if w == target]
        if not target_indices:
            continue

        alignment = process_words(ref_norm, hyp_norm).alignments[0]

        for target_idx in target_indices:
            total += 1
            for chunk in alignment:
                if chunk.ref_start_idx <= target_idx < chunk.ref_end_idx:
                    if chunk.type == "equal":
                        correct += 1
                    elif chunk.type == "delete":
                        deleted += 1
                    elif chunk.type == "substitute":
                        offset = target_idx - chunk.ref_start_idx
                        predicted = hyp_words[chunk.hyp_start_idx + offset]
                        if predicted == context:
                            to_context += 1
                        else:
                            to_other += 1
                    break

    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    return correct / total, to_context / total, to_other / total, deleted / total


def evaluate(results_path, out_path):
    samples = load_results(results_path)

    references = [normalizer(s["reference"]) for s in samples]

    results = {"n_samples": len(samples), "conditions": {}}
    for condition in CONDITIONS:
        hypotheses = [normalizer(s["predicted"][condition]) for s in samples]
        target_rate, context_rate, neither_rate = word_match_rates(samples, condition)
        tgt_correct, tgt_to_context, tgt_to_other, tgt_deleted = compute_target_alignment_stats(samples, condition)
        results["conditions"][condition] = {
            "wer":                    round(wer(references, hypotheses), 4),
            "cer":                    round(cer(references, hypotheses), 4),
            "background_wer":         round(compute_background_wer(samples, condition), 4),
            "target_word_rate":       round(target_rate, 4),
            "context_word_rate":      round(context_rate, 4),
            "neither_rate":           round(neither_rate, 4),
            "target_correct":         round(tgt_correct, 4),
            "target_to_context":      round(tgt_to_context, 4),
            "target_to_other":        round(tgt_to_other, 4),
            "target_deleted":         round(tgt_deleted, 4),
        }

    print(json.dumps(results, indent=2))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    txt_path = out_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Samples: {results['n_samples']}\n\n")
        f.write(f"{'Condition':<25} {'WER':>8} {'CER':>8} {'bg-WER':>8} {'tgt %':>7} {'distr %':>8} {'neither %':>10}"
                f" {'tgt correct':>12} {'tgt->distr':>11} {'tgt->other':>11} {'tgt del':>8}\n")
        sep = "-" * 118 + "\n"
        f.write(sep)

        CONDITION_GROUPS = [
            ["no_context"],
            ["word_context", "word_target"],
            ["sentence_context", "sentences_5_context", "sentences_10_context"],
            ["sentence_target", "sentences_5_target", "sentences_10_target"],
            ["sentences_2_mixed", "sentences_5_mixed", "sentences_10_mixed"],
        ]
        printed = []
        for group in CONDITION_GROUPS:
            for condition in group:
                m = results["conditions"].get(condition)
                if m is None:
                    continue
                f.write(
                    f"{condition:<25} {m['wer']:>8.3f} {m['cer']:>8.3f} {m['background_wer']:>8.3f}"
                    f" {m['target_word_rate']:>6.1%} {m['context_word_rate']:>7.1%} {m['neither_rate']:>9.1%}"
                    f" {m['target_correct']:>11.1%} {m['target_to_context']:>10.1%}"
                    f" {m['target_to_other']:>10.1%} {m['target_deleted']:>7.1%}\n"
                )
                printed.append(condition)
            f.write(sep)
        # print any conditions not covered by the groups above
        for condition, m in results["conditions"].items():
            if condition not in printed:
                f.write(
                    f"{condition:<25} {m['wer']:>8.3f} {m['cer']:>8.3f} {m['background_wer']:>8.3f}"
                    f" {m['target_word_rate']:>6.1%} {m['context_word_rate']:>7.1%} {m['neither_rate']:>9.1%}"
                    f" {m['target_correct']:>11.1%} {m['target_to_context']:>10.1%}"
                    f" {m['target_to_other']:>10.1%} {m['target_deleted']:>7.1%}\n"
                )
        f.write("\n")
        f.write("Column legend:\n")
        f.write("  WER          - Word Error Rate on the full transcription\n")
        f.write("  CER          - Character Error Rate on the full transcription\n")
        f.write("  bg-WER       - Background WER: target word masked in reference, both target+context\n")
        f.write("                 masked in hypothesis. Isolates errors unrelated to the target word.\n")
        f.write("  tgt %        - % of samples where the target word appears anywhere in the hypothesis\n")
        f.write("  distractor % - % of samples where the distractor word appears anywhere in the hypothesis\n")
        f.write("  neither %    - % of samples where neither target nor distractor word appears in the hypothesis\n")
        f.write("  tgt correct  - % of target word positions (by alignment) transcribed correctly\n")
        f.write("  tgt->distr   - % of target word positions substituted with the distractor word (privacy leak signal)\n")
        f.write("  tgt->other   - % of target word positions substituted with an unrelated word\n")
        f.write("  tgt del      - % of target word positions deleted (not transcribed at all)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True, help="Path to output JSONL file")
    parser.add_argument("--out_folder", default="generated_eval", help="Folder to save evaluation results")
    args = parser.parse_args()

    # Strip leading generated_output*/ so paths from generated_output_finetuned/
    # end up cleanly under out_folder just like those from generated_output/
    relative = re.sub(r'^generated_output[^/]*/', '', args.results_path)
    out_path = os.path.join(args.out_folder, relative.replace(".jsonl", ".json"))

    evaluate(args.results_path, out_path)
