"""Compatibility entry point; the research inference workflow is in infer.py."""

from infer import arguments, main, run_evaluation, fit_calibration, apply_calibration

if __name__ == "__main__":
    main()
