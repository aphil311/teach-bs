import string

import torch
from scipy.spatial.distance import cosine
from simalign import SentenceAligner
from transformers import AutoModel, AutoTokenizer

# setup global variables on import (bad practice, but whatever)
# --------------------------------------------------------------

aligner = SentenceAligner(model="distilbert-base-multilingual-cased", layer=6)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")


def accuracy(src_sentence: str, trg_sentence: str) -> dict:
    """
    Calculate the accuracy of a translation by comparing the source and target
    sentences.

    Parameters:
        src_sentence (str): The source sentence.
        trg_sentence (str): The target sentence.

    Returns:
        dict: A dictionary containing the accuracy score and errors.
    """
    # Preprocess both sentences
    src_sentence = __preprocess_text(src_sentence)
    trg_sentence = __preprocess_text(trg_sentence)

    r = __get_alignment_score(src_sentence, trg_sentence)
    score = __get_bertscore(src_sentence, trg_sentence)

    res = {"score": __bertscore_to_percentage(score), "errors": r}
    return res


def __preprocess_text(text: str) -> str:
    """
    Remove punctuation and convert text to lowercase.

    Parameters:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text


def __get_bertscore(src_sentence: str, trg_sentence: str) -> float:
    """
    Get the BERTScore between two sentences.

    Parameters:
        src_sentence (str): The source sentence.
        trg_sentence (str): The target sentence.

    Returns:
        float: The BERTScore.
    """
    # Tokenize and generate embeddings
    inputs_src = tokenizer(
        src_sentence, return_tensors="pt", padding=True, truncation=True
    )
    inputs_trg = tokenizer(
        trg_sentence, return_tensors="pt", padding=True, truncation=True
    )

    with torch.no_grad():
        outputs_src = model(**inputs_src)
        outputs_trg = model(**inputs_trg)

    # Get sentence embeddings by averaging token embeddings (from last hidden state)
    src_embedding = torch.mean(outputs_src.last_hidden_state, dim=1).squeeze().numpy()
    trg_embedding = torch.mean(outputs_trg.last_hidden_state, dim=1).squeeze().numpy()

    # Calculate cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(src_embedding, trg_embedding)

    return similarity


def __bertscore_to_percentage(similarity: float) -> float:
    """
    Convert the BERTScore cosine similarity to a percentage score (0-100).

    Parameters:
        similarity (float): The cosine similarity from BERTScore.

    Returns:
        int: A score from 0 to 100.
    """
    # Scale the similarity score from [-1, 1] range to [0, 100] (rarely negative)
    scaled_score = max(((similarity) / 2) * 100, 0)
    return round(scaled_score, 2)


def __get_alignment_score(src_sentence: str, trg_sentence: str) -> list:
    """
    Get the alignment score between two sentences.

    Parameters:
        src_sentence (str): The source sentence.
        trg_sentence (str): The target sentence.

    Returns:
        list: Mistranslations
    """
    src_list = src_sentence.split()
    trg_list = trg_sentence.split()

    # The output is a dictionary with different matching methods.
    # Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
    alignments = aligner.get_word_aligns(src_list, trg_list)

    src_aligns = {x[0] for x in alignments["inter"]}
    trg_aligns = {x[1] for x in alignments["inter"]}

    mistranslations = []
    for i in range(len(src_list)):
        if i not in src_aligns:
            mistranslations.append(
                {
                    "start": i,
                    "end": i,
                    "message": f"Word {src_list[i]} possibly mistranslated or omitted",
                }
            )

    for i in range(len(trg_list)):
        if i not in trg_aligns:
            mistranslations.append(
                {
                    "start": i,
                    "end": i,
                    "message": f"Word {trg_list[i]} possibly mistranslated or added erroneously",
                }
            )

    return mistranslations
