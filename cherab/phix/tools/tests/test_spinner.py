from time import sleep

import pytest

from cherab.phix.tools.spinner import DummySpinner, Spinner


class TestSpinner:
    @pytest.mark.parametrize("spinner", [Spinner, DummySpinner])
    def test_spinner(self, spinner):
        with spinner():
            sleep(1.0)

    @pytest.mark.parametrize(["spinner", "text"], [(Spinner, "test"), (DummySpinner, "test")])
    def test_spinner_with_text(self, spinner, text):
        with spinner(text=text):
            sleep(1.0)

    @pytest.mark.parametrize("spinner", [Spinner, DummySpinner])
    def test_spinner_decorator(self, spinner):
        @spinner(text="test")
        def test():
            sleep(1.0)

        test()

    @pytest.mark.parametrize("spinner", [Spinner, DummySpinner])
    def test_spinner_change_text(self, spinner):
        with spinner(text="test") as sp:
            sleep(0.5)
            assert sp.text == "test"
            sp.text = "test2"
            sleep(0.5)
            assert sp.text == "test2"

    @pytest.mark.parametrize("spinner", [Spinner, DummySpinner])
    def test_spinner_custom_frames(self, spinner):
        with spinner(frames=["-\\|/"], interval=0.05):
            sleep(0.5)

    @pytest.mark.parametrize("spinner", [Spinner, DummySpinner])
    def test_spinner_writing(self, spinner):
        with spinner() as sp:
            sleep(0.5)
            sp.write("> image 1 download complete")
            sleep(0.5)
            sp.ok()
