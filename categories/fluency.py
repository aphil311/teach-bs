import language_tool_python
import numpy as np
import spacy
import torch
import wordfreq
from transformers import AutoModelForMaskedLM, AutoTokenizer

# setup global variables on import (bad practice, but whatever)
# --------------------------------------------------------------

# grammar checker
tool = language_tool_python.LanguageTool("en-US")

# masked language model and tokenizer from huggingface
model_name = "distilbert-base-multilingual-cased"
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)  # tokenizer

# spacy model for parsing
nlp = spacy.load("en_core_web_sm")


def __get_rarity(word: str, lang: str = "en") -> float:
    """
    Returns the rarity of a word in the given language. word_freq retuns a value
    between 0 and 1, where 1 is the most common word. Therefore, taking the log results
    in a value between 0 (log 1 = 0) and -27.63 (log 1e-12). We then negate it so super
    rare words have a high score and common words have a low score.

    Parameters:
        word (str): The word to check.
        lang (str): The language to check. Default is "en".

    Returns:
        float: The rarity of the word.
    """
    return -np.log(wordfreq.word_frequency(word, lang) + 1e-12)


def __produce_groupings(offset_mapping: list, input_ids: list) -> list:
    """
    Produce groupings of tokens that are part of the same word.

    Parameters:
        offset_mapping (list): The offset mapping of the tokens.
        input_ids (list): The input ids of the tokens.

    Returns:
        list: A list of groupings of tokens.
    """
    # Produce groupings of tokens that are part of the same word
    res = []
    current_group = []
    prev_end = None
    for i, (start, end) in enumerate(offset_mapping):
        if input_ids[i] in tokenizer.all_special_ids:
            continue  # skip special tokens like [CLS] and [SEP]
        if prev_end is not None and start > prev_end:
            # Word boundary detected → start new group
            res.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)
        prev_end = end
    # Append final group
    if current_group:
        res.append(current_group)

    return res


def pseudo_perplexity(text: str, threshold: int = 4, max_len: int = 128) -> dict:
    """
    Calculate the pseudo-perplexity of a text using a masked language model. Return all
    words that exceed a threshold of "adjusted awkwardness". The threshold is a measure
    in terms of log probability of the word.

    Parameters:
        text (str): The text to check.
        threshold (float): The threshold for awkwardness. Default is 4.
        max_len (int): The maximum length of the text. Default is 128.

    Returns:
        dict: A dictionary containing the score and errors.
    """

    # Tokenize the text and produce groupings
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoding["input_ids"][0]
    offset_mapping = encoding["offset_mapping"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    word_groups = __produce_groupings(offset_mapping, input_ids)

    # Calculate the loss for each word group
    loss_values = []
    for group in word_groups:
        # Skip special tokens (CLS and SEP)
        if group[0] == 0 or group[-1] == len(input_ids) - 1:
            continue

        # Mask the word group
        masked = input_ids.clone()
        for i in group:
            masked[i] = tokenizer.mask_token_id

        # Get the model output distribution
        with torch.no_grad():
            outputs = model(masked.unsqueeze(0))
            logits = outputs.logits[0]

        log_probs = []
        for i in group:
            # Get the probability of the true token
            probs = torch.softmax(logits[i], dim=-1)
            true_token_id = input_ids[i].item()
            prob = probs[true_token_id].item()
            # Append the loss of the true token
            log_probs.append(np.log(prob + 1e-12))

        # Calculate the loss for the entire word group
        word_loss = -np.sum(log_probs) / len(log_probs)
        # Adjust the loss based on the rarity of the word
        word = tokenizer.decode(input_ids[group[0]])
        word_loss -= 0.6 * __get_rarity(
            word
        )  # subtract rarity (rare words reduce loss)
        loss_values.append(word_loss)

    # Structure the results for output
    average_loss = np.mean(loss_values)

    errors = []
    for i, l in enumerate(loss_values):
        if l < threshold:
            continue
        errors.append(
            {
                "start": i,
                "end": i,
                "message": f"Adjusted liklihood {round(l, 2)} over threshold {threshold} for word {text.split()[i]}",
            }
        )

    res = {"score": __fluency_score(average_loss), "errors": errors}

    return res


def __fluency_score(
    loss: float, midpoint: float = 5.0, steepness: float = 0.3
) -> float:
    """
    Transform the loss into a score from 0 to 100. Steepness controls how quickly the
    score drops as loss increases and midpoint controls the loss at which the score is
    50.

    Parameters:
        loss (float): The loss to transform.
        midpoint (float): The loss at which the score is 50. Default is 5.
        steepness (float): The steepness of the curve. Default is 0.3.

    Returns:
        float: The score from 0 to 100.
    """
    score = 100 / (1 + np.exp(steepness * (loss - midpoint)))
    return round(score, 2)


def grammar_errors(text: str) -> dict:
    """
    Check the grammar of a text using a grammar checker and a structural grammar check.

    Parameters:
        text (str): The text to check.

    Returns:
        dict: A dictionary containing the score and errors.
    """
    matches = tool.check(text)

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
    for e in struct_err:
        r.append(e)

    grammar_score = len(r) / len(text.split())

    res = {"score": __grammar_score_from_prob(grammar_score), "errors": r}

    return res


def __grammar_score_from_prob(error_ratio: float) -> float:
    """
    Transform the number of errors divided by words into a score from 0 to 100.

    Parameters:
        error_ratio (float): The ratio of errors to words.

    Returns:
        float: The score from 0 to 100.
    """
    score = 100 * (1 - error_ratio)
    return round(score, 2)


def __check_structural_grammar(text: str) -> list:
    """
    Check the structural grammar of a text using spaCy.

    Parameters:
        text (str): The text to check.

    Returns:
        list: A list of structural grammar errors.
    """
    doc = nlp(text)
    issues = []

    # 1. Missing main verb (ROOT)
    root_verbs = [
        tok for tok in doc if tok.dep_ == "ROOT" and tok.pos_ in {"VERB", "AUX"}
    ]
    if not root_verbs:
        root_root = [tok for tok in doc if tok.dep_ == "ROOT"]
        token = root_root[0] if root_root else doc[0]
        issues.append(
            {
                "start": token.i,
                "end": token.i + 1,
                "message": "Sentence is missing a main verb (no ROOT verb).",
            }
        )

    # 2. Verb(s) present but no subject
    verbs = [tok for tok in doc if tok.pos_ in {"VERB", "AUX"}]
    subjects = [tok for tok in doc if tok.dep_ in {"nsubj", "nsubjpass"}]
    if verbs and not subjects:
        for verb in verbs:
            issues.append(
                {
                    "start": verb.i,
                    "end": verb.i + 1,
                    "message": "Sentence has verb(s) but no subject (possible fragment).",
                }
            )

    # 3. Dangling prepositions
    for tok in doc:
        if tok.pos_ == "ADP" and len(list(tok.children)) == 0:
            issues.append(
                {
                    "start": tok.i,
                    "end": tok.i + 1,
                    "message": f"Dangling preposition '{tok.text}' (no object or complement).",
                }
            )

    # 4. Noun pile-up (no verbs, all tokens are nominal)
    if not any(tok.pos_ in {"VERB", "AUX"} for tok in doc) and all(
        tok.pos_ in {"NOUN", "PROPN", "ADJ", "DET", "NUM"}
        for tok in doc
        if tok.is_alpha
    ):
        token = doc[0]
        issues.append(
            {
                "start": token.i,
                "end": token.i + 1,
                "message": "Sentence lacks a verb or any verbal structure (nominal phrase pile-up).",
            }
        )

    # 5. Multiple ROOTs (possible run-on)
    root_count = sum(1 for tok in doc if tok.dep_ == "ROOT")
    if root_count > 1:
        for tok in doc:
            if tok.dep_ == "ROOT":
                issues.append(
                    {
                        "start": tok.i,
                        "end": tok.i + 1,
                        "message": "Sentence has multiple ROOTs — possible run-on sentence.",
                    }
                )

    return issues


# Unit tests can go here eventually
def main():
    pass


if __name__ == "__main__":
    main()
