import os
import subprocess
import tempfile

import numpy as np
from modular_splicing.dataset.additional_data import AdditionalData
from modular_splicing.utils.construct import construct
from modular_splicing.utils.sequence_utils import draw_bases


from permacache import permacache, stable_hash, drop_if_equal

error = """
If you are seeing this message, it means that you have not installed the
RNAFold library.

Follow the instructions here:
    https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html

This uses the CLI interface, since the python interface does not appear to be working.
"""


@permacache(
    "substructrue/rna_fold/run_rna_fold_probabilities_3",
    key_function=dict(sequences=stable_hash, max_distance=drop_if_equal(float("inf"))),
    parallel=["sequences"],
)
def run_rna_fold_probabilities(sequences, max_distance=float("inf")):
    """
    Run RNAFold on a list of sequences, and return the probability that each base
    is paired with another base.

    Arguments
    ---------
    sequences : list of str
        The sequences to run RNAFold on.
    max_distance : int
        The maximum distance between two bases to consider. If two bases are
        further apart than this, they are not considered to be paired.

    Returns
    -------
    list of np.ndarray of equal length to sequences
        The probability that each base is paired with another base.
    """
    fasta = "\n".join(
        x for i, seq in enumerate(sequences) for x in [">seq{}".format(i), seq]
    )
    with tempfile.TemporaryDirectory() as dir, tempfile.NamedTemporaryFile() as fp:
        fp.write(fasta.encode("utf-8"))
        fp.flush()
        try:
            subprocess.check_call(
                ["RNAfold", "-p", "--noPS"],
                cwd=dir,
                stdin=open(fp.name),
                stdout=open(os.devnull, "w"),
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(error)
        results = []
        for i, seq in enumerate(sequences):
            with open(f"{dir}/seq{i}_dp.ps") as f:
                result = list(f)
            result = result[
                result.index("%start of base pair probability data\n") + 1 :
            ]
            matrix = np.zeros((len(seq), len(seq)))
            for line in result:
                line = line.strip()
                if line == "showpage":
                    break
                first, second, amount, ubox = line.split()
                if ubox != "ubox":
                    assert ubox == "lbox"
                    break
                first, second = int(first) - 1, int(second) - 1
                if abs(first - second) > max_distance:
                    continue
                amount = float(amount)
                matrix[first, second] = matrix[second, first] = amount**2
            results.append(matrix.sum(0).astype(np.float16))
        return results


@permacache(
    "substructrue/rna_fold/run_on_entire_sequence_probabilities_2",
    key_function=dict(x=stable_hash, max_distance=drop_if_equal(float("inf"))),
)
def run_on_entire_sequence_probabilities(x, *, sl, cl, max_distance=float("inf")):
    """
    Wrapper around `run_on_entire_sequence_probabilities_all_values` that only
    returns the consensus result.
    """
    _, _, consensus = run_on_entire_sequence_probabilities_all_values(
        x, sl=sl, cl=cl, max_distance=max_distance
    )
    return consensus


def run_on_entire_sequence_probabilities_all_values(x, *, sl, cl, max_distance):
    """
    See `run_on_entire_sequence` for the meaning of the arguments and return
        values.

    Specifically, runs the operation `run_rna_fold_probabilities` on the entire
    sequence, and returns the results. Also concatenates the last result value.
    """

    def process(chunks_with_context):
        results = [
            x
            for x in run_rna_fold_probabilities(
                chunks_with_context, max_distance=max_distance
            )
        ]
        return results

    originals, ranges, results = run_on_entire_sequence_generic(
        x, sl=sl, cl=cl, process=process
    )
    return originals, ranges, np.concatenate(results)


def run_on_entire_sequence_generic(x, *, sl, cl, process):
    """
    Runs the given function on chunks of length `sl` from the original sequence,
        providing a context window of length `cl/2` on either side of the chunk, if
        available. Discards the outputs from the context window.

    Arguments
    ---------
    x : str
        The sequence to run the function on.
    sl : int
        The length of the chunks to run the function on.
    cl : int
        The length of the context window to provide to the function.
    process: function (list of str) -> list of np.ndarray of equal length to list of str

    Returns (originals, ranges, results)
    -------
    originals : list of ndarray
        The original results from the function.
    ranges : dict with keys
        - chunk_ranges: the ranges of the chunks in the original string, excluding the context
        - chunk_contexts: the amount of context provided to each chunk
        - chunk_ranges_with_context: the ranges of the chunks in the original string, including the context
    results : list of ndarray
        The results from the function, with the context window removed.
    """
    seq = draw_bases(x)
    assert len(seq) % sl == 0
    chunk_ranges = [(i, i + sl) for i in range(0, len(seq), sl)]
    chunk_ranges_with_context = [
        (max(0, s - cl), min(e + cl, len(seq))) for s, e in chunk_ranges
    ]
    chunk_contexts = [
        (s - swc, ewc - e)
        for (s, e), (swc, ewc) in zip(chunk_ranges, chunk_ranges_with_context)
    ]
    chunks_with_context = [seq[s:e] for s, e in chunk_ranges_with_context]
    results = process(chunks_with_context)
    new_results = [x[sc : len(x) - ec] for x, (sc, ec) in zip(results, chunk_contexts)]
    return (
        results,
        dict(
            chunk_ranges=chunk_ranges,
            chunk_contexts=chunk_contexts,
            chunk_ranges_with_context=chunk_ranges_with_context,
        ),
        new_results,
    )


class SubstructureProbabilityInformationAdditionalData(AdditionalData):
    """
    Wrapper around the `run_on_entire_sequence_probabilities`

    Also handles preprocessing of the input sequence.

    Arguments:
    sl: sequence length
    cl: context length
    max_distance: maximum distance to consider
    preprocess_spec: preprocessing specification
        either "identity" or "swap_ac"
        identity means no preprocessing
        swap_ac means swapping A and C, to produce the control
    """

    def __init__(
        self,
        *,
        sl,
        cl,
        max_distance=float("inf"),
        preprocess_spec=dict(type="identity"),
    ):
        self.sl = sl
        self.cl = cl
        self.max_distance = max_distance
        self.preprocess_spec = preprocess_spec

    def compute_additional_input(self, original_input, path, i, j):
        x = original_input
        x = construct(
            dict(
                identity=lambda x: x,
                swap_ac=self.swap_ac,
            ),
            self.preprocess_spec,
            x=x,
        )
        assert len(x.shape) == 2
        result = run_on_entire_sequence_probabilities(
            x, sl=self.sl, cl=self.cl, max_distance=self.max_distance
        )

        return result.astype(np.float32)[:, None]

    def swap_ac(self, x):
        x = x.copy()
        a, c = x[:, 0], x[:, 1]
        x = x.copy()
        x[:, 0], x[:, 1] = c, a
        return x
