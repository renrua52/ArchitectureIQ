from tools.evaluate_kb_ab import legacy_v2_solver_prompt


def test_legacy_v2_solver_prompt_preserves_original_claim_format() -> None:
    prompt = legacy_v2_solver_prompt(
        "Which candidate wins?",
        [
            {
                "id": "K0007",
                "text": "Adaptive optimization converges faster under short budgets.",
                "support_count": 3,
                "failure_count": 2,
                "credit": 1,
            }
        ],
    )

    assert (
        "- K0007: Adaptive optimization converges faster under short budgets. "
        "(successful uses=3)"
    ) in prompt
    assert "credit=" not in prompt
    assert "failed uses=" not in prompt
    assert "<question>\nWhich candidate wins?\n</question>" in prompt
