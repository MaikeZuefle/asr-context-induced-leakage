import argparse
import json
import os

from jiwer import cer, wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ExpandCommonEnglishContractions

normalizer = Compose([
    ToLowerCase(),
    ExpandCommonEnglishContractions(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])

CONDITIONS = ["no_context", "with_context_word", "with_context_sentence"]


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def contains_word(text, word):
    return word.lower() in normalizer(text).split()


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


def evaluate(results_path, out_path):
    samples = load_results(results_path)

    references = [normalizer(s["reference"]) for s in samples]

    results = {"n_samples": len(samples), "conditions": {}}
    for condition in CONDITIONS:
        hypotheses = [normalizer(s["predicted"][condition]) for s in samples]
        target_rate, context_rate, neither_rate = word_match_rates(samples, condition)
        results["conditions"][condition] = {
            "wer": round(wer(references, hypotheses), 4),
            "cer": round(cer(references, hypotheses), 4),
            "target_word_rate": round(target_rate, 4),
            "context_word_rate": round(context_rate, 4),
            "neither_rate": round(neither_rate, 4),
        }

    print(json.dumps(results, indent=2))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    txt_path = out_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Samples: {results['n_samples']}\n\n")
        f.write(f"{'Condition':<25} {'WER':>8} {'CER':>8} {'target_word %':>14} {'context_word %':>15} {'neither %':>10}\n")
        f.write("-" * 82 + "\n")
        for condition, m in results["conditions"].items():
            f.write(f"{condition:<25} {m['wer']:>8.3f} {m['cer']:>8.3f} {m['target_word_rate']:>13.1%} {m['context_word_rate']:>14.1%} {m['neither_rate']:>9.1%}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True, help="Path to output JSONL file")
    parser.add_argument("--out_folder", default="generated_eval", help="Folder to save evaluation results")
    args = parser.parse_args()

    relative = os.path.relpath(args.results_path, "generated_output")
    out_path = os.path.join(args.out_folder, relative.replace(".jsonl", ".json"))

    evaluate(args.results_path, out_path)
