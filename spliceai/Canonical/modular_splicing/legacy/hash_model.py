from permacache import stable_hash


def hash_model(m):
    """
    Produces a hash of the model's state dict, along with some of its fields.

    This is a legacy function, and should not be used for new code. Instead use
        stable_hash directly on m. This function is only used for compatibility
        with old permacaches to avoid recomputing them.
    """
    if m is None or getattr(m, "_use_stable_hash_directly", False):
        return stable_hash(m)
    else:
        return stable_hash(
            [
                getattr(m, "thresholds", None),
                m.state_dict(),
                *([] if not hasattr(m, "do_log_softmax") else [m.do_log_softmax]),
                *(
                    []
                    if not hasattr(m, "keep_maximal_width")
                    else [m.keep_maximal_width]
                ),
                *([] if not hasattr(m, "version") else [m.version]),
            ],
            fast_bytes=False,
        )
