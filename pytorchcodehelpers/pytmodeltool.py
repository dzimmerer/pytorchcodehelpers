import sys

import os
import torch

from pytorchcodehelpers.pytorchsize import AbstrModule

if __name__ == '__main__':

    import_str = sys.argv[1]
    import_str = "from " + import_str + " import *"

    exec(import_str)

    module_str = " ".join(sys.argv[2:])
    module_inst = eval(module_str)

    mega_mod = AbstrModule.from_model(module_inst, name="InspectedModule")

    inpt_size = input("Give the input size (eg. 1 3 128 128, if non given will do nothing) : ")

    if inpt_size:
        inpt_size = tuple(map(int, inpt_size.split(" ")))

    input_sample = torch.autograd.Variable(torch.randn(inpt_size))
    mega_mod.set_sizes(input_sample)
    mega_mod.print_sizes()

    print("\nDone.")
