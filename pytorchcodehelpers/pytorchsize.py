import inspect
import re

import torch
import torch.autograd
import torch.nn
from collections import defaultdict


class InspectNet(object):
    @staticmethod
    def get_ircemental_name(var_base_name, var_name_cntr, var_name_set):
        # assert var_base_name in var_name_cntr

        var_name = var_base_name + str(var_name_cntr[var_base_name])
        while var_name in var_name_set:
            var_name_cntr[var_base_name] += 1
            var_name = var_base_name + str(var_name_cntr[var_base_name])

        return var_name

    def __init__(self, net_fn=None, input_size=None, name="NewModule", pre_exec_str =""):

        self.pre_exec_str = pre_exec_str

        self.name = name
        self.size_list = []
        self.init_list = []
        self.fwrd_list = []

        self.fn_param_dict = dict()

        self.input_size = input_size

        if net_fn is not None and input_size is not None:
            self.inspect_net(net_fn, input_size)

    def inspect_net(self, net_fn, input_size=None):

        exec(self.pre_exec_str)

        if input_size is None:
            input_size = self.input_size

        assert input_size is not None

        ### Get stoage vars
        size_list = []
        init_list = []
        fwrd_list = []

        var_name_set = set()
        var_name_cntr = defaultdict(int)

        ### Prepare input
        prev_var_name = "inpt"
        last_tensor = torch.autograd.Variable(torch.randn(input_size))

        size_list.append((prev_var_name, tuple(last_tensor.size())))
        print('%-10s' % prev_var_name, " : ", tuple(last_tensor.size()))
        print("")

        ### Eval function signature
        if inspect.isfunction(net_fn):
            fn_params = inspect.signature(net_fn).parameters

            for param_name, param_expression in fn_params.items():
                if "=" in str(param_expression):
                    exec(str(param_expression))
                self.fn_param_dict[str(param_name)] = str(param_expression)

                ### Preprocess lines
            lines = inspect.getsourcelines(net_fn)[0]
        else:
            if isinstance(net_fn, str):
                lines = net_fn.split("\n")
            elif isinstance(net_fn, list):
                lines = net_fn
            else:
                raise ValueError(
                    "net_fn has to be given as a function object or a list or sting describing the function")

            # Get to def or first expression
            start_ind = 0
            for line in lines:
                if line.strip() == "" or line.strip().startswith("#"):
                    start_ind += 1
                else:
                    break
            lines = lines[start_ind:]

            ### Get parameters from function definition
            if "def" in lines[0]:

                def_line = lines[0]

                fn_args = re.sub(r'.*\(', '', def_line)
                fn_args = re.sub(r'\).*', '', fn_args)

                splitted_args = fn_args.split(",")

                open_brackets = 0
                expression = ""
                for split_arg in splitted_args:

                    if split_arg == "self":
                        continue

                    brack_count = split_arg.count("(") - split_arg.count(")")

                    open_brackets += brack_count
                    expression = expression + split_arg

                    if open_brackets == 0:
                        param_name = expression.split("=")[0].strip()
                        self.fn_param_dict[param_name] = expression.strip()
                        exec(expression.strip())

                        expression = ""

                lines = lines[1:]

        lines = [el.strip() for el in lines]
        lines = [el for el in lines if not el.startswith("super") and not el.startswith("return")]
        lines = [el.replace("self.", "") for el in lines]
        lines = [el.replace("self,", "") for el in lines]
        lines = [el.replace("self", "") for el in lines]

        for line in lines:

            # print(line)

            if not line:
                init_list.append("")
                fwrd_list.append("")
                print("")
                continue
            elif line.startswith("#"):
                init_list.append(line)
                # print(line)
                continue
            elif line.startswith("\"\"\""):
                init_list.append(line)
                # print(line)
                continue

            ## assign variable
            if "=" in line:
                line_split = line.split("=")

                var_name = line_split[0].strip()
                expression = " = ".join(line_split[1:])

                # = in parentheses --> normal expression
                if var_name.count("(") % 2 != 0:
                    var_name = None
                    expression = line

            ## normal expression
            else:

                var_name = None
                expression = line

            var = eval(expression)

            ### Expression did "nothing"
            if var is None and var_name is None:
                continue
            ## Or returned None
            elif var is None and var_name is not None:
                exec(var_name + " = None")
                var_name_set.add(var_name)
                init_list.append(var_name + " = " + expression)
                continue

            ### Expression was a assignation of NN Module
            if isinstance(var, torch.nn.Module):

                ### Get unique variable name
                if var_name is None:
                    var_base_name = type(var).__name__.lower()
                    var_name = InspectNet.get_ircemental_name(var_base_name, var_name_cntr, var_name_set)

                var_name_set.add(var_name)

                ### Calc new tensor and assign to variable
                last_tensor = var(last_tensor)
                exec(var_name + " = last_tensor")

                init_list.append("self.m_" + var_name + " = " + expression)
                fwrd_list.append(var_name + " = self.m_" + var_name + "(" + prev_var_name + ")")

            ### Expression returned a new variable
            elif isinstance(var, torch.autograd.Variable):
                last_tensor = var

                if var_name is None:
                    var_base_name = "var_" + ''.join(e for e in expression.split("(")[0] if e.isalnum())
                    var_name = InspectNet.get_ircemental_name(var_base_name, var_name_cntr, var_name_set)

                var_name_set.add(var_name)
                exec(var_name + " = last_tensor")

                fwrd_list.append(var_name + " = " + expression)

            prev_var_name = var_name

            print('%-10s' % var_name, " : ", tuple(last_tensor.size()))
            size_list.append((var_name, tuple(last_tensor.size())))

        self.size_list = size_list
        self.init_list = init_list
        self.fwrd_list = fwrd_list

        return init_list, fwrd_list, size_list

    def get_init_func_str(self):
        """Returns a valid init function of the Module"""
        assert len(self.init_list) > 0

        method_body_str = "\n\t".join(self.init_list)

        param_list = []

        for param_name, param_expression in self.fn_param_dict.items():
            if param_name != "self" and param_name in method_body_str:
                param_list.append(param_expression)

        param_str = ", ".join(param_list)

        super_str = "super(" + self.name + ", self).__init__()\n\t"

        init_method = "def __init__(self, " + param_str + "):\n\t" + super_str + method_body_str
        return init_method

    def get_fwrd_func_str(self):
        """Returns a valid forward function of the Module"""
        assert len(self.fwrd_list) > 0

        ### method body
        last_name = self.fwrd_list[-1].split("=")[0].strip()
        return_str = "\n\treturn " + last_name
        method_body_str = "\n\t".join(self.fwrd_list) + return_str

        ### function parameters
        param_list = []

        for param_name, param_expression in self.fn_param_dict.items():
            if param_name != "self" and param_name in method_body_str:
                param_list.append(param_expression)

        param_str = ", ".join(param_list)

        fwrd_method = "def forward(self, inpt, " + param_str + "):\n\t" + method_body_str

        return fwrd_method

    def print_sizes(self):

        for var_name, var_size in self.size_list:
            print('%-10s' % var_name, " : ", var_size)

    def get_class_str(self):

        head = "class " + self.name + "(torch.nn.Module):\n"

        init_str = self.get_init_func_str().replace("\t", "\t\t")
        fwrd_str = self.get_fwrd_func_str().replace("\t", "\t\t")

        complete_str = head + "\n\t" + init_str + "\n\n\t" + fwrd_str + "\n"

        return complete_str


class AbstrModule():
    def __init__(self, name="", desc="", cls_name="", python_module="", module=None):

        self.name = name
        self.desc = desc
        self.cls_name = cls_name
        self.python_module = python_module

        self.module = module

        self.submodules = []

        self.input_size = None
        self.output_size = None
        self.hook_handle = None

    def register_hook(self):

        fwrd_hook = AbstrModule.get_fwrd_hook(self)
        self.hook_handle = self.module.register_forward_hook(fwrd_hook)

        for sub_mod in self.submodules:
            sub_mod.register_hook()

    def remove_hook(self):

        if self.hook_handle is not None:
            self.hook_handle.remove()

        self.hook_handle = None
        for sub_mod in self.submodules:
            sub_mod.remove_hook()

    def set_sizes(self, inpt):

        self.register_hook()
        self.module(inpt)
        self.register_hook()

    def print_sizes(self, prefix_str=""):

        print(prefix_str + " (in)  %-15s" % self.cls_name + " : " + str(self.input_size))

        if len(self.submodules) > 0:
            for sub_mod in self.submodules:
                sub_mod.print_sizes(prefix_str=prefix_str + "  ")

        print(prefix_str + " (out) %-15s" % self.cls_name + " : " + str(self.output_size))

    def get_flat_str(self, with_python_module=True):
        if len(self.submodules) > 0:
            strngs = []
            for sub_mod in self.submodules:
                strngs.append(sub_mod.get_flat_str(with_python_module=with_python_module))
            return "\n".join(strngs)
        else:
            if with_python_module:
                return self.python_module + "." + self.desc
            else:
                return self.desc

    @staticmethod
    def get_fwrd_hook(m):
        def frwd_hook(module, input, output):
            m.input_size = tuple(input[0].size())
            m.output_size = tuple(output[0].size())
            return None

        return frwd_hook

    @staticmethod
    def from_model(mod, name=""):
        m = AbstrModule(name=name, desc=repr(mod), cls_name=mod.__class__.__name__, module=mod,
                        python_module=mod.__module__)
        # print(m.cls_name)
        for key, module in mod._modules.items():
            m_child = AbstrModule.from_model(module, name=key)
            m.submodules.append(m_child)
        return m
