"""Compatibility entry point; the research training workflow is in train.py."""

from train import arguments, main, run_training, save_checkpoint, validate

if __name__ == "__main__":
    main()
