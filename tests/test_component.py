# test for pseudo-frozen dataclass Component working correctly
from dataclasses import FrozenInstanceError
from inspect import signature

import pytest
from cosy import Component


@pytest.fixture
def component_no_arguments():
    return Component("", lambda: "0")


@pytest.fixture
def component_with_five_arguments():
    return Component("", lambda _, __, ___, ____, _____: "5")


def test_component_frozen(component_no_arguments) -> None:
    with pytest.raises(FrozenInstanceError):
        component_no_arguments.interpretation = lambda: ""
    with pytest.raises(FrozenInstanceError):
        component_no_arguments.name = ""


def test_wrapper_updated(component_no_arguments, component_with_five_arguments) -> None:
    assert len(list(signature(component_no_arguments).parameters.values())) == 0
    assert len(list(signature(component_with_five_arguments).parameters.values())) == 5


def test_call_components(component_no_arguments, component_with_five_arguments) -> None:
    assert component_no_arguments() == "0"
    assert component_with_five_arguments(None, None, None, None, None) == "5"
