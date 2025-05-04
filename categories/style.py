from transformers import pipeline

pipe = pipeline(
    "text-classification", model="LenDigLearn/formality-classifier-mdeberta-v3-base"
)

formality_score_map = {
    "formal": {"formal": 58, "informal": 0, "neutral": 22},
    "informal": {"formal": 0, "informal": 86, "neutral": 9.7},
    "neutral": {"formal": 20, "informal": 5.1, "neutral": 86},
}


def formality(src_sentence: str, trg_sentence: str) -> dict:
    """
    Evaluate how well the formality of source (German) sentence is
    in translation (English).  Scores are normalized so that the best
    possible match per source‐label is 100.

    Returns:
        {
          "raw_score": float,        # the value from formality_score_map
          "normalized": float,       # raw_score / max_row * 100
          "src_label": str,
          "trg_label": str
        }
    """
    # classify source & target
    src_label = pipe(src_sentence)[0]["label"].lower()
    trg_label = pipe(trg_sentence)[0]["label"].lower()

    # get raw score from the map
    row = formality_score_map.get(src_label, {})
    raw = row.get(trg_label, 0.0)

    # normalize by that row's max
    max_possible = max(row.values()) if row else 1.0
    normalized = (raw / max_possible) * 100

    return {
        "raw_score": raw,
        "normalized": round(normalized, 2),
        "src_label": src_label,
        "trg_label": trg_label,
    }
