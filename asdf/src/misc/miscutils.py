# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import TextIO
import torch

def dual_print(print_file: TextIO, text: str):
    print(text)
    print_file.write(text + '\n')
    print_file.flush()
