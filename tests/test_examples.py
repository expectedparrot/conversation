import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    ["examples.car_buying", "examples.mug_negotiation", "examples.chips"],
)
def test_example_import_does_not_run_inference(module_name, monkeypatch):
    from conversation import ConversationList

    monkeypatch.setattr(
        ConversationList,
        "run",
        lambda self, **kwargs: pytest.fail("example started inference during import"),
    )

    importlib.import_module(module_name)
