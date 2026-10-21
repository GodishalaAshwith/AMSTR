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
    parser.add_argument(
        '--eval',
        action='store_true',
        default=True,
        help='Whether to perform evaluation in train',
    )
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    trainer = Trainer(cfg,
                      mode='train_eval' if FLAGS['eval'] else 'train',
                      task='rec')
    trainer.train()


if __name__ == '__main__':
    main()
