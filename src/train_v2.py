"""Run the shared training pipeline on separately versioned advanced features."""
import argparse
from pathlib import Path

import train
import predict_v2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    train.OUT = Path(__file__).resolve().parents[1] / 'outputs_v2'
    if args.mode == 'train':
        train.train()
    else:
        predict_v2.evidence()
        predict_v2.assemble(train.OUT/'pair_model.joblib', 'submission_advanced.csv')
