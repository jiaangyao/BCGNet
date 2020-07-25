import os

from pathlib import Path
from config import get_config
from session import DefaultSession

d_root = Path(os.getcwd())
cfg = get_config(d_root / 'config' / 'default_config.yaml')
cfg.d_root = Path(os.getcwd())
cfg.num_epochs = 1

str_sub = 'sub11'
vec_idx_run = [1, 2, 3, 4, 5]

s1 = DefaultSession(str_sub, vec_idx_run, str_arch='gru_arch_001', random_seed=1997, verbose=2, overwrite=False,
                    cv_mode=False, cfg=cfg)

s1.load_all_dataset()
s1.prepare_training()
s1.train()
s1.clean()
s1.evaluate(mode='test')

s1.save_model()
s1.save_data()

print('Done')
