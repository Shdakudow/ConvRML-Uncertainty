"""Same protocol as RML, isolated DiffuserCam paths and calibration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import large_interval_study as study

ROOT=Path('results/diffuser_intervals_60k_5k_5k_20260928')
DATA=Path('data/diffuser_intervals_60k_5k_5k_20260928')
CALIBRATION=Path('data/diffuser_intervals_calibration_20260928')


def main():
    study.ROOT=ROOT
    study.DATA=DATA
    study.CALIBRATION=CALIBRATION
    study.main()


if __name__=='__main__':main()
