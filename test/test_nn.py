# TODO: import dependencies and write unit tests below

import numpy as np
from nn.nn import NeuralNetwork
from nn.preprocess import one_hot_encode_seqs, sample_seqs

def test_single_forward():
    """Test the forward pass of a single neuron
    """
    x = np.array([1, 2, 3])
    w = np.array([0.5, 0.5, 0.5])
    b = 0.5

    toy_nn = NeuralNetwork(
        nn_arch=[
            {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
        ],
        loss_function="binary_cross_entropy",
        lr=0.01,
        seed=42,
        epochs=1,
        batch_size=1,
    )
    A, Z = toy_nn._single_forward(A_prev=x, W_curr=w, b_curr=b, activation="sigmoid")
    
    assert A.shape == (1,), "Output shape mismatch"
    assert Z.shape == (1,), "Z shape mismatch"
    

def test_forward():
    """Test the forward pass of the neural network
    """
    x = np.array([1, 2, 3])
    w = np.array([0.5, 0.5, 0.5])
    b = 0.5

    toy_nn = NeuralNetwork(
        nn_arch=[
            {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
        ],
        loss_function="binary_cross_entropy",
        lr=0.01,
        seed=42,
        epochs=1,
        batch_size=1,
    )

    out, cache = toy_nn.forward(x)

    assert out.shape == (1,1), f"Expected output shape (1,1), got {out.shape}"
    assert cache is not None, "Cache should not be None"

    assert 'A0' in cache, "Cache should contain A0"
    assert 'Z1' in cache, "Cache should contain Z1"
    assert 'A1' in cache, "Cache should contain A1"

def test_single_backprop():
    """Test the backpropagation of a single neuron
    """
    pass

def test_predict():
    """Test the prediction of the neural network
    """
    x = np.array([1, 2, 3])

    toy_nn = NeuralNetwork(
        nn_arch=[
            {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
        ],
        loss_function="binary_cross_entropy",
        lr=0.01,
        seed=42,
        epochs=1,
        batch_size=1,
    )

    pred = toy_nn.predict(x)

    assert pred.shape == (1,1), f"Expected prediction shape (1,1), got {pred.shape}"
    assert pred[0][0] >= 0 and pred[0][0] <= 1, "Prediction should be between 0 and 1"

def test_binary_cross_entropy():
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    """Test the sampling of sequences
    """
    sequences = ["ATG", "CGA", "TAC"]
    labels = [0, 1, 0]
    sampled_seqs, sampled_labels = sample_seqs(sequences, labels)

    assert sampled_labels.count(0) == sampled_labels.count(1), "Sampled labels should be balanced"
    assert len(sampled_seqs) == len(sampled_labels), "Sampled sequences and labels should have the same length"


def test_one_hot_encode_seqs():
    """Test the one hot encoding of sequences
    """
    sequences = ["ATGN"]
    one_hot = one_hot_encode_seqs(sequences)

    assert one_hot.shape == (1, 4 * 4), f"Expected shape (1, 4 * 4), got {one_hot.shape}"
    assert np.all(one_hot[0, :4] == [1, 0, 0, 0]), "First nucleotide should be A"
    assert np.all(one_hot[0, -4:] == [0, 0, 0, 0]), "Last nucleotide should be N"
