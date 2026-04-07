from openrecall.itir_client import compare_observations_with_policy_hint


def test_compare_observations_with_policy_hint_round_trips_verified_result(monkeypatch):
    def fake_safe_call_tool(name, payload):
        assert name == "itir.compare_observations"
        assert "left" in payload and "right" in payload
        return {
            "decision": "verified",
            "result": {"similarity": 0.73},
            "status_explanation": {
                "status_scope": "security",
                "status_value": "allowed",
                "status_bucket": "allow",
                "primary_reason_code": "verified",
                "reason_codes": ["verified"],
            },
            "policy_outcomes": ["verified"],
        }

    monkeypatch.setattr("openrecall.itir_client.safe_call_tool", fake_safe_call_tool)
    payload = compare_observations_with_policy_hint(
        {"source_system": "openrecall", "text": "left"},
        {"source_system": "openrecall", "text": "right"},
    )

    assert payload["decision"] == "verified"
    assert payload["result"]["similarity"] == 0.73
    assert payload["policy_hint"]["primary_reason_code"] == "verified"
    assert payload["status_explanation"]["status_scope"] == "security"


def test_compare_observations_with_policy_hint_preserves_reject_without_raising(monkeypatch):
    def fake_safe_call_tool(name, payload):
        return {
            "decision": "rejected",
            "result": None,
            "status_explanation": {
                "status_scope": "security",
                "status_value": "rejected",
                "status_bucket": "review",
                "primary_reason_code": "sensitive_data_request",
                "reason_codes": ["sensitive_data_request"],
            },
            "policy_outcomes": [],
        }

    monkeypatch.setattr("openrecall.itir_client.safe_call_tool", fake_safe_call_tool)
    payload = compare_observations_with_policy_hint(
        {"source_system": "openrecall", "text": "left"},
        {"source_system": "openrecall", "text": "right"},
    )

    assert payload["decision"] == "rejected"
    assert payload["result"] is None
    assert payload["policy_hint"]["decision"] == "rejected"
    assert payload["policy_hint"]["primary_reason_code"] == "sensitive_data_request"
