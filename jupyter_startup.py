"""
Jupyter kernel startup script.

Deploy to: /etc/jupyter/kernel_startup.py
Wire up via jupyter_notebook_config.py:
    c.IPKernelApp.exec_files = ["/etc/jupyter/kernel_startup.py"]

Provides:
  - DistillationModeController  – session-level mode tracker
  - %set_distillation_mode on|off  – IPython line magic
  - %distillation_status           – IPython line magic
  - DISTILLATION global variable injected into kernel namespace
"""
import logging
import os

logger = logging.getLogger(__name__)


class DistillationModeController:
    """
    Tracks the distillation mode for the current Jupyter kernel session.

    The mode is kept in sync with the DISTILLATION environment variable so
    that load_config() and the import guard always see the current value.
    """

    def __init__(self) -> None:
        raw = os.getenv("DISTILLATION", "off")
        self.mode = self._validate(raw)
        # Ensure env var is normalised from the start
        os.environ["DISTILLATION"] = self.mode

    @staticmethod
    def _validate(value: str) -> str:
        v = value.lower().strip()
        return v if v in ("on", "off") else "off"

    def set(self, value: str) -> str:
        """
        Update mode, sync env var, and notify frozen_layer_modules (if loaded).

        Returns:
            The validated mode string ("on" or "off").
        """
        self.mode = self._validate(value)
        os.environ["DISTILLATION"] = self.mode
        try:
            import frozen_layer_modules as flm
            flm.set_mode(self.mode)
        except ImportError:
            pass
        return self.mode

    def status(self) -> str:
        """Return a human-readable status string."""
        try:
            from frozen_layer_modules.config import load_config
            cfg = load_config()
            return (
                f"Distillation Mode:    {self.mode.upper()}\n"
                f"Drift Weight:         {cfg.drift_weight}\n"
                f"Restoration Factor:   {cfg.restoration_factor}\n"
                f"Divergence Threshold: {cfg.divergence_threshold}\n"
                f"Divergence Weight:    {cfg.divergence_weight}"
            )
        except Exception:
            return f"Distillation Mode: {self.mode.upper()}"


# ---------------------------------------------------------------------------
# Module-level controller instance
# ---------------------------------------------------------------------------
_controller = DistillationModeController()

# ---------------------------------------------------------------------------
# IPython magic registration (silently skipped outside Jupyter)
# ---------------------------------------------------------------------------
try:
    from IPython.core.magic import register_line_magic
    from IPython import get_ipython as _get_ipython

    @register_line_magic
    def set_distillation_mode(line: str) -> None:
        """
        Toggle distillation mode without restarting the kernel.

        Usage:
            %set_distillation_mode on
            %set_distillation_mode off
        """
        arg = line.strip().lower()
        if arg not in ("on", "off"):
            print(
                f"Invalid value '{arg}'. "
                "Valid values: 'on', 'off'. "
                "Usage: %set_distillation_mode on|off"
            )
            return
        mode = _controller.set(arg)
        print(f"Distillation mode: {mode.upper()}")

    @register_line_magic
    def distillation_status(line: str) -> None:
        """
        Print current distillation mode and configuration.

        Usage:
            %distillation_status
        """
        print(_controller.status())

    # Inject globals into the kernel namespace
    _ip = _get_ipython()
    if _ip is not None:
        _ip.push({
            "DISTILLATION": _controller.mode,
            "distillation_controller": _controller,
        })

except (ImportError, RuntimeError):
    # Running outside a Jupyter kernel – magic registration is a no-op
    pass
