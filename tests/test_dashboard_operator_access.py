import pytest

from alpha.live.dashboard_v3.app import create_app


class ForbiddenProvider:
    def get_summary_dict(self):
        raise AssertionError("Request unexpectedly accessed operational data")


@pytest.mark.parametrize("access_token_str", [None, "short", "obsolete-operator-credential-123456"])
def test_health_available_without_login_even_with_stale_env(monkeypatch, access_token_str):
    if access_token_str is None:
        monkeypatch.delenv("ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR", raising=False)
    else:
        monkeypatch.setenv("ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR", access_token_str)
    response_obj = create_app(ForbiddenProvider()).test_client().get("/healthz")
    assert response_obj.status_code == 200
    assert "WWW-Authenticate" not in response_obj.headers
    assert response_obj.headers["Cache-Control"] == "no-store"
    assert response_obj.headers["X-Frame-Options"] == "DENY"


def test_missing_client_registry_is_not_a_password_error(monkeypatch):
    monkeypatch.delenv("ALPHA_CLIENT_REPORTING_CONFIG_PATH_STR", raising=False)
    response_obj = create_app(ForbiddenProvider()).test_client().get("/clients", follow_redirects=True)
    assert response_obj.status_code == 200
    assert "WWW-Authenticate" not in response_obj.headers
    assert "Operator access must be configured" not in response_obj.get_data(as_text=True)


@pytest.mark.parametrize("forwarded_proto_str", ["http", "https"])
def test_remote_plain_http_rejected_without_trusting_proxy_headers(forwarded_proto_str):
    response_obj = create_app(ForbiddenProvider()).test_client().get(
        "/healthz", headers={"X-Forwarded-Proto": forwarded_proto_str},
        environ_overrides={"REMOTE_ADDR": "192.0.2.10"},
    )
    assert response_obj.status_code == 426


def test_remote_https_does_not_require_application_login():
    response_obj = create_app(ForbiddenProvider()).test_client().get(
        "/healthz", base_url="https://localhost",
        environ_overrides={"REMOTE_ADDR": "192.0.2.10"},
    )
    assert response_obj.status_code == 200


def test_no_login_does_not_bypass_read_only_actions():
    client_obj = create_app(ForbiddenProvider(), read_only_bool=True).test_client()
    for path_str in ("/api/pods/any/actions/tick", "/api/pods/any/manual-order", "/api/pods/any/diff/run"):
        assert client_obj.post(path_str).status_code == 403
    for path_str in ("/api/action-token", "/fragments/action-preview/any/tick",
                     "/fragments/manual-order-ticket/any", "/api/pods/any/trade-sheet"):
        assert client_obj.get(path_str).status_code == 403


@pytest.mark.parametrize("path_str", ["/fragments/command-catalog/ambiguous", "/api/pods/ambiguous/trade-sheet"])
@pytest.mark.parametrize("read_only_bool,status_int", [(False, 409), (True, 403)])
def test_ambiguous_target_is_controlled_and_never_reaches_export_or_actions(path_str, read_only_bool, status_int):
    class AmbiguousProvider(ForbiddenProvider):
        def get_target_for_pod(self, pod_id_str):
            assert status_int == 409, "Read-only must reject before lookup"
            raise ValueError("PRIVATE_PATH duplicate target")

        def export_trade_sheet_path_str(self, target_obj):
            pytest.fail("Ambiguous target must not export a file")

    app_obj = create_app(AmbiguousProvider(), read_only_bool=read_only_bool)
    response_obj = app_obj.test_client().get(path_str)
    assert response_obj.status_code == status_int
    assert "PRIVATE_PATH" not in response_obj.get_data(as_text=True)
    if status_int == 409:
        assert "target_unavailable" in response_obj.get_data(as_text=True)
