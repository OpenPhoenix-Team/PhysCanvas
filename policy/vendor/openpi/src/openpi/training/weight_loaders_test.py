import numpy as np

from openpi.training import weight_loaders


def test_tactile_partial_merge_only_initializes_tactile_tree():
    loaded = {"old": {"weight": np.ones((2,), dtype=np.float32)}}
    reference = {
        "old": {"weight": np.zeros((2,), dtype=np.float32)},
        "tactile_encoder": {"gate": np.asarray(0.01, dtype=np.float32)},
        "unexpected_new_module": {"weight": np.zeros((1,), dtype=np.float32)},
    }

    merged = weight_loaders._merge_params(  # noqa: SLF001 - directly tests the strict merge primitive.
        loaded,
        reference,
        missing_regex=r".*(?:lora|tactile_encoder).*",
    )

    assert "old" in merged
    assert "tactile_encoder" in merged
    assert "unexpected_new_module" not in merged
