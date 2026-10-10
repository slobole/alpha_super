"""The Tools page says what it does: production only copies commands."""

from alpha.live.dashboard_v4.demo import create_demo_app


def test_tools_verdict_is_honest_and_order_capable_commands_are_marked():
    html_str = create_demo_app().test_client().get("/tools?pod=demo_1_0").get_data(as_text=True)
    assert "Copy a command and run it in the VPS terminal. Nothing runs from this page." in html_str
    assert "Copy or run an operator command." not in html_str
    assert html_str.count("ACTIVE · CAN SEND ORDERS") == 5  # tick, run_once, serve, submit_vplan, manual_order
    simulated_str = create_demo_app(demo_tools_bool=True).test_client().get("/tools").get_data(as_text=True)
    assert "Copy or run an operator command." in simulated_str
