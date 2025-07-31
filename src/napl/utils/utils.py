import os, sys, yaml, json, torch, random
import numpy as np
import importlib.util
import napl

from collections import OrderedDict
from yamlordereddictloader import SafeDumper
from yamlordereddictloader import SafeLoader
from loguru import logger
from dataclasses import dataclass


class bcolors:
    """
    default color palette
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    red = '#EF553B'
    orange = '#E58606'
    yellow = '#FABD2F'
    green = '#9CC424'
    cyan = '#6FD19F'
    blue = '#FABD2F'
    purple = '#AB82FF'
    gray = '#CCCCCC'
    gray2 = '#999999'
    gray3 = '#666666'
    gray4 = '#333333'

    ResetAll = '\033[0m'
    Bold       = '\033[1m'
    Dim        = '\033[2m'
    Underlined = '\033[4m'
    Blink      = '\033[5m'
    Reverse    = '\033[7m'
    Hidden     = '\033[8m'

    ResetBold       = '\033[21m'
    ResetDim        = '\033[22m'
    ResetUnderlined = '\033[24m'
    ResetBlink      = '\033[25m'
    ResetReverse    = '\033[27m'
    ResetHidden     = '\033[28m'

    Default      = '\033[39m'
    Black        = '\033[30m'
    Red          = '\033[31m'
    Green        = '\033[32m'
    Yellow       = '\033[33m'
    Blue         = '\033[34m'
    Magenta      = '\033[35m'
    Cyan         = '\033[36m'
    LightGray    = '\033[37m'
    DarkGray     = '\033[90m'
    LightRed     = '\033[91m'
    LightGreen   = '\033[92m'
    LightYellow  = '\033[93m'
    LightBlue    = '\033[94m'
    LightMagenta = '\033[95m'
    LightCyan    = '\033[96m'
    White        = '\033[97m'

    BackgroundDefault      = '\033[49m'
    BackgroundBlack        = '\033[40m'
    BackgroundRed          = '\033[41m'
    BackgroundGreen        = '\033[42m'
    BackgroundYellow       = '\033[43m'
    BackgroundBlue         = '\033[44m'
    BackgroundMagenta      = '\033[45m'
    BackgroundCyan         = '\033[46m'
    BackgroundLightGray    = '\033[47m'
    BackgroundDarkGray     = '\033[100m'
    BackgroundLightRed     = '\033[101m'
    BackgroundLightGreen   = '\033[102m'
    BackgroundLightYellow  = '\033[103m'
    BackgroundLightBlue    = '\033[104m'
    BackgroundLightMagenta = '\033[105m'
    BackgroundLightCyan    = '\033[106m'
    BackgroundWhite        = '\033[107m'


def strip_list(input_list: list) -> list:
    """
    strip leading and trailing spaces for each list item
    """
    l = []

    for e in input_list:
        e = e.strip()
        if e != '' and e != ' ':
            l.append(e)

    return l


def check_type(input, type):
    """
    check whether input is the required type
    """
    assert isinstance(input, type), logger.error('Invalid input type')
    

def check_file_list(file_list: list):
    """
    check whether all files in the list exist
    """
    for file in file_list:
        assert os.path.exists(file), logger.error('No file: ' + file)


def clean_file_list(file_list: list):
    """
    delete files in the list, if they exist
    """
    for file in file_list:
        if os.path.exists(file):
            logger.warning('Delete file: ' + file)
            os.remove(file)


def create_dir(directory):
    """
    Checks the existence of a directory, if does not exist, create a new one
    :param directory: path to directory under concern
    :return: None
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.success('Create directory: ' + directory)
    except OSError:
        logger.error('Create directory: ' +  directory)
        sys.exit()
    

def create_subdir(path: str, subdir_list: list):
    for subdir in subdir_list:
        subdir_path = os.path.join(path, subdir.strip('/'))
        if not os.path.exists(subdir_path):
            create_dir(subdir_path)

    
def read_yaml(file):
    return yaml.load(open(file), Loader=SafeLoader)


def write_yaml(file, content):
    """
    if file exists at filepath, overwite the file, if not, create a new file
    :param filepath: string that specifies the destination file path
    :param content: yaml string that needs to be written to the destination file
    :return: None
    """
    if os.path.exists(file):
        os.remove(file)
    create_dir(os.path.dirname(file))
    out_file = open(file, 'a')
    out_file.write(yaml.dump( content, default_flow_style= False, Dumper=SafeDumper))


def check_repeated_key(full_dict: OrderedDict, key:str, val: OrderedDict):
    key_index = list(full_dict.keys()).index(key)
    key_list = list(full_dict.keys())[0 : key_index]
    for key in key_list:
        if full_dict[key] == val:
            return True, key
    return False, None
    

# The following interpolate_oneD_linear and interpolate_oneD_quadratic are adapted from accelergy
# ===============================================================
# useful helper functions that are commonly used in estimators
# ===============================================================
def interpolate_oneD_linear(desired_x, known):
    """
    utility function that performs 1D linear interpolation with a known energy value
    :param desired_x: integer value of the desired attribute/argument
    :param known: list of dictionary [{x: <value>, y: <energy>}]
    :return energy value with desired attribute/argument
    """
    # assume E = ax + c where x is a hardware attribute
    ordered_list = []
    if known[1]['x'] < known[0]['x']:
        ordered_list.append(known[1])
        ordered_list.append(known[0])
    else:
        ordered_list = known

    slope = (known[1]['y'] - known[0]['y']) / (known[1]['x'] - known[0]['x'])
    desired_energy = slope * (desired_x - ordered_list[0]['x']) + ordered_list[0]['y']
    return desired_energy


def interpolate_oneD_quadratic(desired_x, known):
    """
    utility function that performs 1D linear interpolation with a known energy value
    :param desired_x: integer value of the desired attribute/argument
    :param known: list of dictionary [{x: <value>, y: <energy>}]
    :return energy value with desired attribute/argument
    """
    # assume E = ax^2 + c where x is a hardware attribute
    ordered_list = []
    if known[1]['x'] < known[0]['x']:
        ordered_list.append(known[1])
        ordered_list.append(known[0])
    else:
        ordered_list = known

    slope = (known[1]['y'] - known[0]['y']) / (known[1]['x']**2 - known[0]['x']**2)
    desired_energy = slope * (desired_x**2 - ordered_list[0]['x']**2) + ordered_list[0]['y']
    return desired_energy


def get_input_tuple(input, size=2):
    if isinstance(input, tuple):
        assert len(input) == size, logger.error('Invalid input size: ' + str(len(input)) + '!=' + str(size))
        return input
    else:
        output = (input, ) * size
        return output


def get_path(path):
    path = os.path.abspath(path)
    path = os.path.realpath(path)
    assert os.path.exists(path), logger.error('Invalid path: ' + path)
    return path


def uniquify_list(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_dict(input_dict: OrderedDict):
    return json.loads(json.dumps(input_dict))


def check_dict_in_list(input_dict, input_list):
    return get_dict(input_dict) in input_list


def check_dict_equal(input_dict0, input_dict1):
    return get_dict(input_dict0) == get_dict(input_dict1)


def get_prod(input_array):
    return np.prod(np.array(input_array))


def check_yaml_header(input_dict: OrderedDict, header: str, yaml_path: str):
    assert header in input_dict.keys(), logger.error(f'Missing header <{header}> in .{header}.yaml at <{yaml_path}>.')


def check_yaml_cfg(input_dict: OrderedDict, key: str, yaml_path: str):
    assert key in input_dict.keys(), logger.error(f'Missing key <{key}> in the configuration at <{yaml_path}>.')


def call_func_from_yaml(yaml_path: str=None, header: str=None, func_name: str=None, py_path: str=None, **kwargs):
    full_path = get_path(yaml_path)
    load_cfg = read_yaml(full_path)

    # check yaml header
    check_yaml_header(load_cfg, header, full_path)

    # get config
    load_cfg = load_cfg[header]

    # func_name has to be specified
    check_yaml_cfg(load_cfg, func_name, full_path)
    func = load_cfg[func_name].lower()

    # find proper func_name to create the header
    dst_file = os.path.join(py_path, func, func + '.py')
    spec = importlib.util.spec_from_file_location(f'create_{header}_with_{func}', dst_file)
    module_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module_py
    spec.loader.exec_module(module_py)

    return module_py.create(load_cfg, **kwargs)


def call_func_from_yaml(yaml_path: str=None, header: str=None, func_name: str=None, py_path: str=None, **kwargs):
    full_path = get_path(yaml_path)
    load_cfg = read_yaml(full_path)

    # check yaml header
    check_yaml_header(load_cfg, header, full_path)

    # get config
    load_cfg = load_cfg[header]

    # func_name has to be specified
    check_yaml_cfg(load_cfg, func_name, full_path)
    func = load_cfg[func_name].lower()

    # find proper func_name to create the header
    dst_file = os.path.join(py_path, func, func + '.py')
    spec = importlib.util.spec_from_file_location(f'create_{header}_with_{func}', dst_file)
    module_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module_py
    spec.loader.exec_module(module_py)

    return module_py.create(load_cfg, **kwargs)


def call_func_from_cfg(cfg: dict, header: str, func_name: str, py_path: str, **kwargs):
    check_yaml_cfg(cfg, func_name, '<in-memory>')
    func = cfg[func_name].lower()

    dst_file = os.path.join(py_path, func, func + '.py')
    spec = importlib.util.spec_from_file_location(f'create_{header}_with_{func}', dst_file)
    module_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module_py
    spec.loader.exec_module(module_py)

    return module_py.create(cfg, **kwargs)


def check_config(config: dict, key_list: list):
    """
    Check if all key in the key_list exists in the config.
    """
    for key in key_list:
        assert key in config, logger.error(f'Missing key <{key}> in the input configuration.')


def check_mode(config: dict):
    """
    Check if polarity is legal.
    """
    polarity = config.get('polarity', None)
    if polarity is not None:
        assert isinstance(polarity, str), logger.error(f'Invalid polarity: <{polarity}>; polarity should be a string.')
        polarity = polarity.lower()
        legal_modes = ['unipolar', 'bipolar']
        assert polarity in legal_modes, logger.error(f'Invalid polarity: <{polarity}>; legal values: <{str(legal_modes)}>.')
    return polarity


def gen_rand_tensor(polarity: str = 'unipolar', shape: tuple = (1,), width: int = 8):
    """
    Generate a random fraction in the range [0, 1).
    """
    prob = torch.rand(shape)
    if polarity == 'unipolar':
        data = prob
        return (data * (2 ** width)).floor() / (2 ** width)
    else:
        data = (prob * 2 - 1)
        return (data * (2 ** (width - 1))).floor() / (2 ** (width - 1))


def gen_arange_tensor(polarity: str = 'unipolar', width: int = 8):
    """
    Generate all fraction in the range [0, 1).
    """
    prob = torch.arange(2 ** width) / (2 ** width)
    if polarity == 'unipolar':
        data = prob
        return (data * (2 ** width)).floor() / (2 ** width)
    else:
        data = (prob * 2 - 1)
        return (data * (2 ** (width - 1))).floor() / (2 ** (width - 1))


def check_name(config: dict):
    """
    Check if name is available.
    """
    name = config.get('name', None)
    if name is not None:
        assert isinstance(name, str), logger.error(f'Invalid name: <{name}>; name should be a string.')
        name = name.lower()
    return name

