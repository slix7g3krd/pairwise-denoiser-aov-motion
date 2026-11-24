import pytest
import graphics.denoiser_aov_motion as mod

CASES = [
    ("aov_align", (1,)),
    ("motion_layers", (1,)),
    ("history_valid", (1,)),
    ("history_reset", (1,)),
    ("frame_align", (1,)),
    ("seed_fixed", (1,)),
    ("history_clip", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("reproject", (1,)),
    ("metrics", (1,)),
    ("teleport", (1,)),
    ("async_result", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
