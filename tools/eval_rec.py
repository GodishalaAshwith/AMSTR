import os
import sys

# NumPy 2.0+ compatibility shim for legacy packages
try:
    import numpy as np
    if not hasattr(np, 'sctypes'):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [getattr(np, t) for t in ['float16', 'float32', 'float64', 'longdouble'] if hasattr(np, t)],
            'complex': [getattr(np, t) for t in ['complex64', 'complex128', 'clongdouble'] if hasattr(np, t)],
            'others': [bool, object, bytes, str, np.void]
        }
except ImportError:
    pass

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    trainer = Trainer(cfg, mode='eval')

    best_model_dict = trainer.status.get('metrics', {})
    trainer.logger.info('metric in ckpt ***************')
    for k, v in best_model_dict.items():
        trainer.logger.info('{}:{}'.format(k, v))

    metric = trainer.eval()

    trainer.logger.info('metric eval ***************')
    for k, v in metric.items():
        trainer.logger.info('{}:{}'.format(k, v))

    output_dir = cfg['Global'].get('output_dir', None)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        res_file = os.path.join(output_dir, 'eval_results.txt')
        with open(res_file, 'w', encoding='utf-8') as f:
            f.write(f"Checkpoint: {cfg['Global'].get('checkpoints')}\n")
            f.write("--- Metrics ---\n")
            for k, v in metric.items():
                f.write(f"{k}: {v}\n")
        trainer.logger.info(f"Evaluation results successfully saved to: {res_file}")


if __name__ == '__main__':
    main()
