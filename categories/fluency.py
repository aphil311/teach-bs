import language_tool_python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import spacy

tool = language_tool_python.LanguageTool('en-US')
model_name="distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

nlp = spacy.load("en_core_web_sm")

def pseudo_perplexity(text, max_len=128):
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
    input_ids = tokenizer.encode(text, return_tensors="pt")[0]

    if len(input_ids) > max_len:
        raise ValueError(f"Input too long for model (>{max_len} tokens).")

    loss_values = []

    for i in range(1, len(input_ids) - 1):  # skip [CLS] and [SEP]
        masked_input = input_ids.clone()
        masked_input[i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input.unsqueeze(0))
            logits = outputs.logits[0, i]
            probs = torch.softmax(logits, dim=-1)

        true_token_id = input_ids[i].item()
        prob_true_token = probs[true_token_id].item()
        log_prob = np.log(prob_true_token + 1e-12)
        loss_values.append(-log_prob)
    
    # get longest sequence of tokens with perplexity over some threshold
    threshold = 12  # Define a perplexity threshold
    longest_start, longest_end = 0, 0
    current_start, current_end = 0, 0
    max_length = 0
    curr_loss = 0

    for i, loss in enumerate(loss_values):
        if loss > threshold:
            if current_start == current_end:  # Start a new sequence
                current_start = i
            current_end = i + 1
            curr_loss = loss
        else:
            if current_end - current_start > max_length:
                longest_start, longest_end = current_start, current_end
                max_length = current_end - current_start
            current_start, current_end = 0, 0

    if current_end - current_start > max_length:  # Check the last sequence
        longest_start, longest_end = current_start, current_end

    longest_sequence = (longest_start, longest_end)

    ppl = np.exp(np.mean(loss_values))

    res = {
        "score": __fluency_score_from_ppl(ppl),
        "errors": [
            {
                "start": longest_sequence[0],
                "end": longest_sequence[1],
                "message": f"Perplexity above threshold: {curr_loss}"
            }
        ]
    }

    return res

def __fluency_score_from_ppl(ppl, midpoint=20, steepness=0.3):
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

def __grammar_score_from_prob(error_ratio, steepness=10):
    """
    Transform the number of errors divided by words into a score from 0 to 100.
    Steepness controls how quickly the score drops as errors increase.
    """
    score = 100 / (1 + np.exp(steepness * error_ratio))
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
