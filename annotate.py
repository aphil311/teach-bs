import json
import sys

from tqdm import tqdm

from categories.accuracy import get_bertscore
from categories.fluency import grammar_errors, pseudo_perplexity


def annotate_entries(entries):
    """
    Add BERTScore, fluency (pseudo-perplexity), and grammar-error scores to each translation pair.

    Args:
        entries (List[Dict]): dicts with "german" and "english" keys.

    Returns:
        The same list, with "bertscore", "fluency_score", and "grammar_score" added to each entry.
    """
    for ex in tqdm(entries):
        src = ex["german"]
        tgt = ex["english"]

        sim = get_bertscore(src, tgt)

        pp = pseudo_perplexity(tgt)

        ge = grammar_errors(tgt)

        # append new fields
        ex["bertscore"] = round(float(sim), 4)
        ex["fluency_score"] = float(pp["score"])
        ex["grammar_score"] = float(ge["score"])

    return entries


def main(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotated = annotate_entries(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    print(f"Annotated {len(annotated)} entries → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python annotate_translations.py  input.json  output.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
