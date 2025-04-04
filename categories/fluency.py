import language_tool_python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import spacy
import wordfreq

tool = language_tool_python.LanguageTool('en-US')
model_name="distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

nlp = spacy.load("en_core_web_sm")

def __get_word_pr_score(word, lang="en") -> list[float]:
    return -np.log(wordfreq.word_frequency(word, lang) + 1e-12)

def pseudo_perplexity(text, threshold=20, max_len=128):
    """
    We want to return
    {
        "score": normalized value from 0 to 100,
        "errors": [
            {
                "start": word index,
                "end": word index,
                "message": "error message"
            }
        ]
    }
    """
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoding["input_ids"][0]
    # print(input_ids)
    offset_mapping = encoding["offset_mapping"][0]
    # print(offset_mapping)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Group token indices by word based on offset mapping
    word_groups = []
    current_group = []

    prev_end = None

    for i, (start, end) in enumerate(offset_mapping):
        if input_ids[i] in tokenizer.all_special_ids:
            continue  # skip special tokens like [CLS] and [SEP]

        if prev_end is not None and start > prev_end:
            # Word boundary detected → start new group
            word_groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)

        prev_end = end

    # Append final group
    if current_group:
        word_groups.append(current_group)

    loss_values = []
    tok_loss = []
    for group in word_groups:
        if group[0] == 0 or group[-1] == len(input_ids) - 1:
            continue  # skip [CLS] and [SEP]

        masked = input_ids.clone()
        for i in group:
            masked[i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked.unsqueeze(0))
            logits = outputs.logits[0]

        log_probs = []
        for i in group:
            probs = torch.softmax(logits[i], dim=-1)
            true_token_id = input_ids[i].item()
            prob = probs[true_token_id].item()
            log_probs.append(np.log(prob + 1e-12))
            tok_loss.append(-np.log(prob + 1e-12))

        word_loss = -np.sum(log_probs) / len(log_probs)
        word = tokenizer.decode(input_ids[group[0]])
        word_loss -= 0.6 * __get_word_pr_score(word)
        loss_values.append(word_loss)
    
    # print(loss_values)

    errors = []
    for i, l in enumerate(loss_values):
        if l < threshold:
            continue
        errors.append({
            "start": i,
            "end": i,
            "message": f"Perplexity {l} over threshold {threshold}"
        })

    # print(tok_loss)
    s_ppl = np.mean(tok_loss)
    # print(s_ppl)

    res = {
        "score": __fluency_score_from_ppl(s_ppl),
        "errors": errors
    }

    return res

def __fluency_score_from_ppl(ppl, midpoint=8, steepness=0.3):
    """
    Use a logistic function to map perplexity to 0–100.
    Midpoint is the PPL where score is 50.
    Steepness controls curve sharpness.
    """
    score = 100 / (1 + np.exp(steepness * (ppl - midpoint)))
    return round(score, 2)

def grammar_errors(text) -> tuple[int, list[str]]:
    """

    Returns
      int: number of grammar errors
      list: grammar errors
        tuple: (start, end, error message)
    """

    matches = tool.check(text)
    grammar_score = len(matches)/len(text.split())

    r = []
    for match in matches:
        words = text.split()
        char_to_word = []
        current_char = 0

        for i, word in enumerate(words):
            for _ in range(len(word)):
                char_to_word.append(i)
            current_char += len(word)
            if current_char < len(text):  # Account for spaces between words
                char_to_word.append(i)
                current_char += 1

        start = char_to_word[match.offset]
        end = char_to_word[match.offset + match.errorLength - 1] + 1
        r.append({"start": start, "end": end, "message": match.message})

    struct_err = __check_structural_grammar(text)
    r.extend(struct_err)

    res = {
        "score": __grammar_score_from_prob(grammar_score),
        "errors": r
    }

    return res

def __grammar_score_from_prob(error_ratio):
    """
    Transform the number of errors divided by words into a score from 0 to 100.
    Steepness controls how quickly the score drops as errors increase.
    """
    score = 100*(1-error_ratio)
    return round(score, 2)


def __check_structural_grammar(text):
    doc = nlp(text)
    issues = []

    # 1. Missing main verb (ROOT)
    root_verbs = [tok for tok in doc if tok.dep_ == "ROOT" and tok.pos_ in {"VERB", "AUX"}]
    if not root_verbs:
        root_root = [tok for tok in doc if tok.dep_ == "ROOT"]
        token = root_root[0] if root_root else doc[0]
        issues.append({
            "start": token.i,
            "end": token.i + 1,
            "message": "Sentence is missing a main verb (no ROOT verb)."
        })

    # 2. Verb(s) present but no subject
    verbs = [tok for tok in doc if tok.pos_ in {"VERB", "AUX"}]
    subjects = [tok for tok in doc if tok.dep_ in {"nsubj", "nsubjpass"}]
    if verbs and not subjects:
        for verb in verbs:
            issues.append({
                "start": verb.i,
                "end": verb.i + 1,
                "message": "Sentence has verb(s) but no subject (possible fragment)."
            })

    # 3. Dangling prepositions
    for tok in doc:
        if tok.pos_ == "ADP" and len(list(tok.children)) == 0:
            issues.append({
                "start": tok.i,
                "end": tok.i + 1,
                "message": f"Dangling preposition '{tok.text}' (no object or complement)."
            })

    # 4. Noun pile-up (no verbs, all tokens are nominal)
    if not any(tok.pos_ in {"VERB", "AUX"} for tok in doc) and \
       all(tok.pos_ in {"NOUN", "PROPN", "ADJ", "DET", "NUM"} for tok in doc if tok.is_alpha):
        token = doc[0]
        issues.append({
            "start": token.i,
            "end": token.i + 1,
            "message": "Sentence lacks a verb or any verbal structure (nominal phrase pile-up)."
        })

    # 5. Multiple ROOTs (possible run-on)
    root_count = sum(1 for tok in doc if tok.dep_ == "ROOT")
    if root_count > 1:
        for tok in doc:
            if tok.dep_ == "ROOT":
                issues.append({
                    "start": tok.i,
                    "end": tok.i + 1,
                    "message": "Sentence has multiple ROOTs — possible run-on sentence."
                })

    return issues
