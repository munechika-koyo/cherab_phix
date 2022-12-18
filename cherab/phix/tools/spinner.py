"""Module offering terminal spinner Spinner implementation is referred by the
`yaspin` package:

https://github.com/pavdmyt/yaspin
"""
from __future__ import annotations

import contextlib
import functools
import sys
import time
from collections.abc import Callable, Iterable
from datetime import timedelta
from itertools import cycle
from multiprocessing import Event, Lock, Process

__all__ = ["Spinner"]


SPINNERS = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]


class Spinner:
    """Implements a context manager that spawns a child process to write
    spinner frames into a tty (stdout) duringcontext execution.

    Parameters
    ----------
    text
        Text to show along with spinner, by default "Loading..."
    interval
        spinners wait time, by default 0.1 sec
    frames
        spinner animated frames, by default ``["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]``
    timer
        Prints a timer showing the elapsed time, by default False
    side
        Place spinner to the right or left end of the text string, by default "left"


    Example
    -------
    In test.py,

    .. code-block:: python

        import time
        from cherab.phix.tools import Spinner


        # Use as a context manager
        with Spinner():
            time.sleep(3.0)

        # Context manager with text
        with Spinner(text="Processing..."):
            time.sleep(3.0)

        # Context manager with custom sequence
        with Spinner(frames="-\\|/", interval=0.05):
            time.sleep(3.0)

        # As decorator
        @Spinner(text="Loading...")
        def foo():
            time.sleep(3.0)
        foo()

        # Context manager writing message
        with Spinner() as sp:
            # task 1
            time.sleep(1.0)
            sp.write("> image 1 download complete")

            # task 2
            time.sleep(2.0)
            sp.write("> image 2 download complete")

            # finalize
            time.sleep(1.0)
            sp.ok("✅")

    Here is the result when the above script is excuted.

    .. image:: ../_static/images/spinner_example.gif
    """

    def __init__(
        self,
        text: str = "Loading...",
        interval: float = 0.1,
        frames: Iterable[str] = SPINNERS,
        timer: bool = False,
        side: str = "left",
    ) -> None:
        self.text = text
        self.interval = interval
        self.frames = frames
        self._cycle = cycle(self._frames)
        self.timer = timer
        self.side = side

        self._start_time = None
        self._stop_time = None
        self._hidden_level = 0
        self._cur_line_len = 0

        self._stop_spin = Event()
        self._hide_spin = Event()
        self._stdout_lock = Lock()
        self._spin_process = Process(target=self._animate, daemon=True)

    # === dunders ==================================================================================
    def __repr__(self) -> str:
        return f"<Spinner object frames={self._frames!s}>"

    def __call__(self, fn) -> Callable:
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return inner

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        # Avoid stop() execution for the 2nd time
        if self._spin_process.is_alive():
            self.stop()
        return False

    # === Properties ===============================================================================
    @property
    def text(self) -> str:
        """Text to show along with spinner."""
        return self._text

    @text.setter
    def text(self, value: str):
        if not isinstance(value, str):
            raise TypeError("text must be a str type.")
        self._text = value

    @property
    def interval(self) -> float:
        """spinners wait time."""
        return self._interval

    @interval.setter
    def interval(self, value):
        if not isinstance(value, float):
            raise TypeError("interval must be a float type.")
        self._interval = value

    @property
    def frames(self) -> Iterable[str]:
        """spinner animated frames."""
        return self._frames

    @frames.setter
    def frames(self, values):
        if not isinstance(values, Iterable):
            for val in values:
                if not isinstance(val, str):
                    raise TypeError("frames must be a Iterable composing strings.")
        self._frames = values

    @property
    def timer(self) -> bool:
        """Prints a timer showing the elapsed time."""
        return self._timer

    @timer.setter
    def timer(self, value):
        if not isinstance(value, bool):
            raise TypeError("timer must be a boolen type.")
        self._timer = value

    @property
    def side(self):
        """Place spinner to the right or left end of the text string."""
        return self._side

    @side.setter
    def side(self, value):
        if value not in {"right", "left"}:
            raise ValueError("side is must be either 'right' or 'left'.")
        self._side = value

    @property
    def elapsed_time(self) -> float:
        """Return calculated elapsed time."""
        if self._start_time is None:
            return 0

        if self._stop_time is None:
            return time.time() - self._start_time

        return self._stop_time - self._start_time

    # === methods ==================================================================================
    def start(self):
        """start spinner process."""
        self._hide_cursor()
        self._start_time = time.time()
        self._stop_time = None
        self._stop_spin.clear()
        self._hide_spin.clear()
        try:
            self._spin_process.start()
        except Exception:
            self._show_cursor()
        return self

    def stop(self):
        """stop spinner process."""
        self._stop_time = time.time()

        if self._spin_process:
            self._stop_spin.set()
            self._spin_process.join()

        self._clear_line()
        self._show_cursor()

    def hide(self):
        """Hide the spinner to allow for custom writing to the terminal."""
        process_is_alive = self._spin_process and self._spin_process.is_alive()

        if process_is_alive and not self._hide_spin.is_set():
            with self._stdout_lock:
                # set the hidden spinner flag
                self._hide_spin.set()
                self._clear_line()

                # flush the stdout buffer so the current line
                # can be rewritten to
                sys.stdout.flush()

    @contextlib.contextmanager
    def hidden(self):
        """Hide the spinner within a block, can be nested."""
        if self._hidden_level == 0:
            self.hide()
        self._hidden_level += 1

        try:
            yield
        finally:
            self._hidden_level -= 1
            if self._hidden_level == 0:
                self.show()

    def show(self):
        """Show the hidden spinner."""
        process_is_alive = self._spin_process and self._spin_process.is_alive()

        if process_is_alive and self._hide_spin.is_set():
            with self._stdout_lock:
                # clear the hidden spinner flag
                self._hide_spin.clear()

                # clear the current line so the spinner is not appended to it
                self._clear_line()

    def write(self, text: str) -> None:
        """Write text in the terminal without breaking the spinner."""
        with self._stdout_lock:
            self._clear_line()

            # Ensure output is Unicode
            assert isinstance(text, str)

            sys.stdout.write(f"{text}\n")
            self._cur_line_len = 0

    def ok(self, text: str = "✅") -> None:
        """Set Ok (success) finalizer to a spinner.

        Parameters
        ----------
        text
            Ok success text, by default "✅"
        """
        _text = text if text else "✅"
        self._freeze(_text)

    def fail(self, text: str = "💥") -> None:
        """Set fail finalizer to a spinner.

        Parameters
        ----------
        text
            fail text, by default "💥"
        """
        _text = text if text else "💥"
        self._freeze(_text)

    def _freeze(self, final_text: str) -> None:
        """Stop spinner, compose last frame and 'freeze' it."""
        self._last_frame = self._compose_out(final_text, mode="last")

        # Should be stopped here, otherwise prints after
        # self._freeze call will mess up the spinner
        self.stop()
        with self._stdout_lock:
            sys.stdout.write(self._last_frame)
            self._cur_line_len = 0

    def _compose_out(self, frame: str, mode: str | None = None):
        # Ensure Unicode input
        assert isinstance(frame, str)

        text = str(self._text)
        assert isinstance(text, str)

        # Colors
        # if self._color_func is not None:
        #     frame = self._color_func(frame)

        # Position
        if self._side == "right":
            frame, text = text, frame

        if self._timer:
            sec, fsec = divmod(round(100 * self.elapsed_time), 100)
            text += f" ({timedelta(seconds=sec)}.{fsec:02.0f})"

        # Mode
        if not mode:
            out = f"\r{frame} {text}"
        else:
            out = f"{frame} {text}\n"

        # Ensure output is Unicode
        assert isinstance(out, str)

        return out

    def _animate(self) -> None:
        """animate spinners in a child process."""
        while not self._stop_spin.is_set():

            if self._hide_spin.is_set():
                # Wait a bit to avoid wasting cycles
                time.sleep(self._interval)
                continue

            spin_frame = next(self._cycle)
            out = self._compose_out(spin_frame)

            self._clear_line()
            sys.stdout.write(out)
            sys.stdout.flush()
            self._cur_line_len = max(self._cur_line_len, len(out))

            time.sleep(self._interval)

    def _clear_line(self) -> None:
        if sys.stdout.isatty():
            # ANSI Control Sequence EL does not work in Jupyter
            sys.stdout.write("\r\033[K")
        else:
            fill = " " * self._cur_line_len
            sys.stdout.write(f"\r{fill}\r")

    @staticmethod
    def _hide_cursor() -> None:
        if sys.stdout.isatty():
            # ANSI Control Sequence DECTCEM 1 does not work in Jupyter
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()

    @staticmethod
    def _show_cursor() -> None:
        if sys.stdout.isatty():
            # ANSI Control Sequence DECTCEM 2 does not work in Jupyter
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()


if __name__ == "__main__":

    @Spinner()
    def test():
        time.sleep(3.0)

    test()

    with Spinner("Loading with context manager..."):
        for i in range(10):
            time.sleep(0.25)
