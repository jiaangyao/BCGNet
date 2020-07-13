import os
from pathlib import Path


def update_init(dest_dir):
    vec_f_file = []

    if os.name == 'nt':
        import win32api, win32con

    for d_file in dest_dir.iterdir():
        f_file = d_file.stem

        if os.name == 'nt':
            attribute = win32api.GetFileAttributes(f_file)
            bool_hidden = attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
        else:
            bool_hidden = f_file.startswith('.')

        if not bool_hidden:
            vec_f_file.append(f_file)

    if "__init__" in vec_f_file:
        vec_f_file.remove("__init__")

    if "__pycache__" in vec_f_file:
        vec_f_file.remove("__pycache__")

    with open(dest_dir / '__init__.py', 'w') as f:
        for f_file in vec_f_file:
            f.write("from models.{} import *\n".format(f_file))

    f.close()


if __name__ == '__main__':
    """ used for debugging """

    curr_dir = Path(os.getcwd())
    update_init(curr_dir)

    print('nothing')