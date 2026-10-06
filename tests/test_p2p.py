"""Portable checks for the pretrained adapter's temporal and control semantics."""
import pytest
import torch
from roblox_learner.p2p import policy_mask, decode_actions, P2PPolicy


def test_policy_mask_prevents_future_leakage_and_preserves_action_history():
    positions = torch.arange(36)
    mask = policy_mask(positions, positions)
    # Frame 1 output can see all frame 0 tokens except its output summary.
    assert mask[15, :12].tolist() == [True, True, True, False] + [True] * 8
    # Frame 1 image sees its own three observation tokens, no current action.
    assert mask[13, 12:24].tolist() == [True] * 3 + [False] * 9
    # Output sees observations and itself; action tokens attend mutually.
    assert mask[15, 12:24].tolist() == [True] * 4 + [False] * 8
    assert mask[16, 12:24].tolist() == [True] * 3 + [False] + [True] * 8
    assert not mask[:24, 24:].any()
    assert not policy_mask(torch.tensor([2412]), torch.tensor([0])).item()


def test_action_decoder_preserves_simultaneous_movement_jump_and_mouse_buttons():
    result = decode_actions([11, 1, 18, 0, 1, 2, 12, 8])
    assert result == {"keys": ["shift", "space", "w"], "buttons": ["left", "right"],
                      "mouse_dx_pixels": 1, "mouse_dy_pixels": 0}
    assert decode_actions([0] * 6 + [11, 8])["keys"] == []


@pytest.mark.parametrize("tokens", [[0] * 7, [20] + [0] * 7, [0] * 7 + [17], [True] + [0] * 7])
def test_invalid_actions_rejected(tokens):
    with pytest.raises(ValueError):
        decode_actions(tokens)


def test_rotary_embedding_matches_complex_number_reference():
    torch.manual_seed(7)
    x = torch.randn(1, 3, 4, 16)
    positions = torch.tensor([7, 8, 9])
    theta = 10000 ** (-torch.arange(0, 16, 2).float() / 16)
    rotations = torch.polar(torch.ones(3, 8), positions[:, None] * theta)
    reference = torch.view_as_real(torch.view_as_complex(x.reshape(1, 3, 4, 8, 2)) * rotations[None, :, None]).flatten(-2)
    torch.testing.assert_close(P2PPolicy.rope(x, positions), reference)


def test_multi_button_controls_hold_then_release_on_focus_loss():
    from test_runtime import FakeDesktop
    from roblox_learner.p2p_play import P2PController
    from roblox_learner.desktop import FocusLost
    backend = FakeDesktop()
    controller = P2PController(backend)
    tokens = [11, 1, 0, 0, 1, 2, 11, 8]
    controller.apply_tokens(tokens)
    controller.apply_tokens(tokens)
    assert backend.events.count(("button", "left", True)) == 1
    assert backend.events.count(("button", "right", True)) == 1
    backend.focused = False
    with pytest.raises(FocusLost):
        controller.apply_tokens(tokens)
    assert not controller.buttons and not controller.held_keys
    assert ("button", "left", False) in backend.events
    assert ("button", "right", False) in backend.events


def test_shadow_trial_captures_and_predicts_without_input(tmp_path):
    from test_runtime import FakeDesktop
    from roblox_learner.p2p_play import run_trial
    class Policy:
        device = "cpu"
        def reset(self): pass
        def predict(self, image): return {"tokens": [11, 1, 0, 0, 1, 2, 11, 8]}
    backend = FakeDesktop()
    report = run_trial(backend, Policy(), tmp_path / "trial", seconds=.12, fps=30)
    assert not backend.events
    assert report["steps"] > 0
    assert report["live_input"] is False and report["outcome"] == "unverified"
    assert (tmp_path / "trial" / "stop.png").is_file()


def test_removed_dummy_queries_cannot_change_first_four_outputs():
    # Every layer uses this mask, so dropping the last eight dummy queries
    # preserves the first four queries by induction through the transformer.
    q, k, v = (torch.randn(1, 2, 12, 8) for _ in range(3))
    mask = policy_mask(torch.arange(12), torch.arange(12))
    full = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    short = torch.nn.functional.scaled_dot_product_attention(q[:, :, :4], k[:, :, :4], v[:, :, :4], attn_mask=mask[:4, :4])
    torch.testing.assert_close(short, full[:, :, :4])
