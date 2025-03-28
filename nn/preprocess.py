# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Separate sequences by their labels
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label]
    neg_seqs = [seq for seq, label in zip(seqs, labels) if not label]

    # Determine the number of samples to draw
    num_samples = max(len(pos_seqs), len(neg_seqs))

    # Sample with replacement to balance the classes
    sampled_pos_seqs = np.random.choice(pos_seqs, num_samples, replace=True).tolist()
    sampled_neg_seqs = np.random.choice(neg_seqs, num_samples, replace=True).tolist()

    # Pad sequences to all be the same length with 'N'
    max_length = max(len(seq) for seq in sampled_pos_seqs + sampled_neg_seqs)
    sampled_pos_seqs = [seq.ljust(max_length, 'N') for seq in sampled_pos_seqs]
    sampled_neg_seqs = [seq.ljust(max_length, 'N') for seq in sampled_neg_seqs]

    # Combine the sampled sequences and labels
    sampled_seqs = sampled_pos_seqs + sampled_neg_seqs
    sampled_labels = [True] * num_samples + [False] * num_samples

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    dict_map = {
        'A': [1, 0, 0, 0], 
        'T': [0, 1, 0, 0], 
        'C': [0, 0, 1, 0], 
        'G': [0, 0, 0, 1], 
    }

    # Initialize an empty list to hold the one-hot encodings
    out = []
    # Iterate through each sequence
    for seq in seq_arr:
        encodings = []
        # For each base in the sequence, append its one-hot encoding
        for base in seq:
            encodings.extend(dict_map.get(base, [0, 0, 0, 0]))
        # Convert the list to a numpy array and return
        out.append(np.array(encodings))

    out = np.array(out)

    return out