import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

import glob

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))


# import all the arch modules
# arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]

# instead of above, do it for all "sub"-file, recursively
arch_filenames = glob.glob("**/*_arch.py", recursive=True)


arch_filenames = [file_name.replace('\\', '/') for file_name in arch_filenames]  # for windows
arch_filenames = [file_name.replace('/', '.').rstrip(".py") for file_name in arch_filenames]   # to module format (convert slash to dots, remove .py)


_arch_modules = [importlib.import_module(file_name) for file_name in arch_filenames]







def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
