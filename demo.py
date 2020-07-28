import os

from pathlib import Path
from config import get_config
from session import DefaultSession

d_root = Path(os.getcwd())
cfg = get_config(d_root / 'config' / 'default_config.yaml')
cfg.d_root = Path(os.getcwd())

cv_mode = True
if cv_mode:
    cfg.num_epochs = 1
    cfg.per_training = 0.6
    cfg.per_valid = 0.2
    cfg.per_test = 0.2

    cfg.d_output = Path('/home/jyao/Downloads/cv_data')
    cfg.d_model = Path('/home/jyao/Downloads/cv_model')

str_sub = 'sub11'
vec_idx_run = [1, 2, 3, 4, 5]

s1 = DefaultSession(str_sub=str_sub, vec_idx_run=vec_idx_run, str_arch='gru_arch_001',
                    random_seed=1997, verbose=2, overwrite=False,
                    cv_mode=cv_mode, cfg=cfg)

s1.load_all_dataset()
s1.prepare_training()
s1.train()
s1.clean()
s1.evaluate(mode='test')

s1.save_model()
s1.save_data()

print('Done')
