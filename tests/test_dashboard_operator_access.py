from base64 import b64encode

import pytest

from alpha.live.dashboard_v3.app import create_app


TEST_ACCESS_STR = "synthetic-test-operator-access-123456"


def auth_headers_dict(password_str=TEST_ACCESS_STR, username_str="operator"):
    return {"Authorization": "Basic " + b64encode(f"{username_str}:{password_str}".encode()).decode()}


class ForbiddenProvider:
    def get_summary_dict(self):
        raise AssertionError("Unauthenticated request accessed operational data")


@pytest.mark.parametrize("path_str", ["/", "/performance", "/clients", "/clients/sample/overview", "/diagnostics", "/api/action-token", "/healthz"])
def test_configured_console_requires_operator_before_any_data_access(path_str):
    app_obj = create_app(ForbiddenProvider(), operator_access_token_str=TEST_ACCESS_STR)
    response_obj = app_obj.test_client().get(path_str)
    assert response_obj.status_code == 401
    assert response_obj.headers["Cache-Control"] == "no-store"
    assert response_obj.headers["X-Frame-Options"] == "DENY"


def test_client_reporting_fails_closed_without_access_configuration(monkeypatch):
    monkeypatch.delenv("ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR", raising=False)
    response_obj = create_app(ForbiddenProvider()).test_client().get("/clients")
    assert response_obj.status_code == 503


@pytest.mark.parametrize("password_str,username_str", [("bad", "operator"), (TEST_ACCESS_STR, "client"), ("סיסמה", "operator")])
def test_wrong_operator_credential_rejected(password_str, username_str):
    response_obj = create_app(ForbiddenProvider(), operator_access_token_str=TEST_ACCESS_STR).test_client().get("/", headers=auth_headers_dict(password_str, username_str))
    assert response_obj.status_code == 401


def test_remote_plain_http_rejected_even_with_valid_credential():
    response_obj = create_app(ForbiddenProvider(), operator_access_token_str=TEST_ACCESS_STR).test_client().get("/", headers=auth_headers_dict(), environ_overrides={"REMOTE_ADDR": "192.0.2.10"})
    assert response_obj.status_code == 426


def test_authentication_does_not_bypass_read_only_actions():
    client_obj = create_app(ForbiddenProvider(), operator_access_token_str=TEST_ACCESS_STR, read_only_bool=True).test_client()
    for path_str in ("/api/pods/any/actions/tick", "/api/pods/any/manual-order", "/api/pods/any/diff/run"):
        response_obj = client_obj.post(path_str, headers=auth_headers_dict())
        assert response_obj.status_code == 403


def test_authenticated_healthz_is_available_without_financial_data():
    response_obj = create_app(ForbiddenProvider(), operator_access_token_str=TEST_ACCESS_STR).test_client().get("/healthz", headers=auth_headers_dict())
    assert response_obj.status_code == 200


def test_weak_operator_credential_rejected():
    with pytest.raises(ValueError, match="24 characters"):
        create_app(ForbiddenProvider(), operator_access_token_str="short")
