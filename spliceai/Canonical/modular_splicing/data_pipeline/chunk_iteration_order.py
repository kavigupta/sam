DATASET_CHUNK_SIZE = 100


def indices_for_chunk(sequence_size, chunk_size, chunk_index):
    """
    Produce a set of indices for a chunk of a dataset.

    Produce extra for the last chunk if necessary, but have the rest be exactly 100.
    """
    if (chunk_index + 1) == sequence_size // chunk_size:
        current_chunk_size = chunk_size + sequence_size % chunk_size
    else:
        current_chunk_size = chunk_size

    return [chunk_index * chunk_size + j for j in range(current_chunk_size)]


def dataset_indices_generator(sequence_size, chunk_size=DATASET_CHUNK_SIZE):
    """
    Produce the indices for all chunks of a dataset.
    """
    if sequence_size // chunk_size == 0:
        yield 0, [i for i in range(sequence_size)]
        return

    for i in range(sequence_size // chunk_size):
        yield i, indices_for_chunk(sequence_size, chunk_size, i)
