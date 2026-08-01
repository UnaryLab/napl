import os
import sys
import yaml
import json
import torch
import math
import numpy as np
import importlib.util

from collections import OrderedDict
from yamlordereddictloader import SafeDumper
from yamlordereddictloader import SafeLoader
from loguru import logger


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
    stripped = []

    for e in input_list:
        e = e.strip()
        if e != '' and e != ' ':
            stripped.append(e)

    return stripped


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
    """Create each named subdirectory below a parent path.

    Args:
        path: Parent directory for the requested subdirectories.
        subdir_list: Subdirectory names. Leading and trailing ``/`` characters
            are ignored.

    Returns:
        ``None``.
    """
    for subdir in subdir_list:
        subdir_path = os.path.join(path, subdir.strip('/'))
        if not os.path.exists(subdir_path):
            create_dir(subdir_path)


def read_yaml(file):
    """Load one YAML file with the ordered safe loader.

    Args:
        file: Path to the YAML file.

    Returns:
        The Python object decoded from the YAML document.
    """
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
    """Find an earlier mapping entry whose value equals a target value.

    Args:
        full_dict: Ordered mapping to search.
        key: Current key, which limits the search to preceding entries.
        val: Value to compare with each preceding entry.

    Returns:
        A ``(repeated, key)`` pair. The key is the first matching earlier key,
        or ``None`` when no match exists.
    """
    key_index = list(full_dict.keys()).index(key)
    key_list = list(full_dict.keys())[0 : key_index]
    for key in key_list:
        if full_dict[key] == val:
            return True, key
    return False, None


def interpolate_oneD_linear(desired_x, known):
    """
    utility function that performs 1D linear interpolation with a known energy value
    :param desired_x: integer value of the desired attribute/argument
    :param known: list of dictionary [{x: <value>, y: <energy>}]
    :return energy value with desired attribute/argument
    """
    # E = ax + c, where x is a hardware attribute.
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
    # E = ax^2 + c, where x is a hardware attribute.
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
    """Normalize a scalar or tuple argument to a fixed-size tuple.

    Args:
        input: Existing tuple or value to repeat.
        size: Required tuple length. Defaults to ``2``.

    Returns:
        The original tuple, or a tuple containing ``input`` repeated ``size``
        times.
    """
    if isinstance(input, tuple):
        assert len(input) == size, logger.error('Invalid input size: ' + str(len(input)) + '!=' + str(size))
        return input
    else:
        output = (input, ) * size
        return output


def get_path(path):
    """Resolve an existing path to its absolute real path.

    Args:
        path: Path to resolve.

    Returns:
        Absolute path with symbolic links resolved.
    """
    path = os.path.abspath(path)
    path = os.path.realpath(path)
    assert os.path.exists(path), logger.error('Invalid path: ' + path)
    return path


def uniquify_list(sequence):
    """Remove duplicate hashable values while preserving their first occurrence.

    Args:
        sequence: Iterable of hashable values.

    Returns:
        List containing each distinct value once in input order.
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_dict(input_dict: OrderedDict):
    """Convert a mapping to JSON-compatible built-in containers.

    Args:
        input_dict: Mapping whose keys and values are JSON serializable.

    Returns:
        Dictionary produced by a JSON encode-and-decode round trip.
    """
    return json.loads(json.dumps(input_dict))


def check_dict_in_list(input_dict, input_list):
    """Check whether a normalized mapping occurs in a list.

    Args:
        input_dict: Mapping to normalize with :func:`get_dict`.
        input_list: Candidate list of dictionaries.

    Returns:
        ``True`` when the normalized mapping is present; otherwise ``False``.
    """
    return get_dict(input_dict) in input_list


def check_dict_equal(input_dict0, input_dict1):
    """Compare two mappings after JSON-compatible normalization.

    Args:
        input_dict0: First mapping.
        input_dict1: Second mapping.

    Returns:
        ``True`` when both normalized dictionaries are equal; otherwise
        ``False``.
    """
    return get_dict(input_dict0) == get_dict(input_dict1)


def get_prod(input_array):
    """Compute the product of values after converting them to a NumPy array.

    Args:
        input_array: Array-like values to multiply.

    Returns:
        NumPy scalar containing the product of all values.
    """
    return np.prod(np.array(input_array))


def check_yaml_header(input_dict: OrderedDict, header: str, yaml_path: str):
    """Require a top-level key in a loaded YAML mapping.

    Args:
        input_dict: Loaded YAML mapping.
        header: Required top-level key.
        yaml_path: Source path included in the error message.

    Returns:
        ``None``.
    """
    assert header in input_dict.keys(), logger.error(f'Missing header <{header}> in .{header}.yaml at <{yaml_path}>.')


def check_yaml_cfg(input_dict: OrderedDict, key: str, yaml_path: str):
    """Require a configuration key in a loaded YAML mapping.

    Args:
        input_dict: Configuration mapping.
        key: Required key.
        yaml_path: Source path included in the error message.

    Returns:
        ``None``.
    """
    assert key in input_dict.keys(), logger.error(f'Missing key <{key}> in the configuration at <{yaml_path}>.')


def call_func_from_yaml(yaml_path: str=None, header: str=None, func_name: str=None, py_path: str=None, **kwargs):
    """Load a YAML configuration and call its selected factory module.

    Args:
        yaml_path: Path to the YAML configuration.
        header: Top-level mapping passed to the factory.
        func_name: Key whose lowercase value names the factory directory.
        py_path: Parent directory containing ``<name>/<name>.py`` factories.
        **kwargs: Extra keyword arguments passed to the factory's ``create()``
            function.

    Returns:
        Value returned by the selected module's ``create()`` function.
    """
    full_path = get_path(yaml_path)
    load_cfg = read_yaml(full_path)

    check_yaml_header(load_cfg, header, full_path)

    load_cfg = load_cfg[header]

    check_yaml_cfg(load_cfg, func_name, full_path)
    func = load_cfg[func_name].lower()

    dst_file = os.path.join(py_path, func, func + '.py')
    spec = importlib.util.spec_from_file_location(f'create_{header}_with_{func}', dst_file)
    module_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module_py
    spec.loader.exec_module(module_py)

    return module_py.create(load_cfg, **kwargs)


def call_func_from_cfg(cfg: dict, header: str, func_name: str, py_path: str, **kwargs):
    """Call a factory module selected by an in-memory configuration.

    Args:
        cfg: Configuration mapping passed to the factory.
        header: Name used to identify the dynamically loaded module.
        func_name: Key whose lowercase value names the factory directory.
        py_path: Parent directory containing ``<name>/<name>.py`` factories.
        **kwargs: Extra keyword arguments passed to the factory's ``create()``
            function.

    Returns:
        Value returned by the selected module's ``create()`` function.
    """
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


def check_polarity(config: dict):
    """
    Check if polarity is legal.
    """
    polarity = config.get('polarity', None)
    if polarity is not None:
        assert isinstance(polarity, str), logger.error(f'Invalid polarity: <{polarity}>; polarity should be a string.')
        polarity = polarity.lower()
        legal_polarity = ['unipolar', 'bipolar']
        assert polarity in legal_polarity, logger.error(f'Invalid polarity: <{polarity}>; legal values: <{str(legal_polarity)}>.')
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


def pow2_lshift(input, shift):
    """
    Power-of-two left shift for float tensors: input * 2**shift.
    Stock torch '<<'/'>>' are integer-only, but unary/fixed-point kernels need shifts
    on float tensors. Use these shims instead of patching the operators. `shift` may be
    an int or a tensor.
    """
    return input * (2.0 ** shift)


def pow2_rshift(input, shift):
    """
    Power-of-two right shift for float tensors: input / 2**shift. See pow2_lshift.
    """
    return input / (2.0 ** shift)


def rshift_offset(input, weight, widthi, widthw, rounding="round", quantilei=1, quantilew=1):
    """
    Dynamic fixed-point scaling: return the right-shift offsets that bring `input` and
    `weight` into a `widthi`/`widthw`-bit range (from their quantile-clipped magnitude),
    plus the output offset that undoes both.
    """
    def _mag(x, q):
        # q=1 bypasses torch.quantile, which rejects tensors larger than 2**24 elements.
        if q == 1:
            return x.abs().max()
        lower = torch.quantile(x, 0.5 + q / 2)
        upper = torch.quantile(x, 0.5 - q / 2)
        return torch.max(lower.abs(), upper.abs())

    with torch.no_grad():
        imax_int = _mag(input, quantilei).log2()
        wmax_int = _mag(weight, quantilew).log2()

        if rounding == "round":
            imax_int = imax_int.round()
            wmax_int = wmax_int.round()
        elif rounding == "floor":
            imax_int = imax_int.floor()
            wmax_int = wmax_int.floor()
        elif rounding == "ceil":
            imax_int = imax_int.ceil()
            wmax_int = wmax_int.ceil()

        # Zero-magnitude operands map log2(0) to a zero offset, keeping results finite.
        imax_int = torch.nan_to_num(imax_int, nan=0.0, neginf=0.0, posinf=0.0)
        wmax_int = torch.nan_to_num(wmax_int, nan=0.0, neginf=0.0, posinf=0.0)

        rshift_i = imax_int - widthi
        rshift_w = wmax_int - widthw
        rshift_o = max(widthi, widthw) - imax_int - wmax_int
        return rshift_i, rshift_w, rshift_o


def num2tuple(num):
    """Return num as a 2-tuple: a scalar becomes (num, num), a tuple passes through."""
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """Spatial (H, W) of a conv2d output."""
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h, w


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    """Padding (as (top,bottom),(left,right)) to map h_w_in to h_w_out."""
    h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation)
    p_h = ((h_w_out[0] - 1) * stride[0] - h_w_in[0] + dilation[0] * (kernel_size[0] - 1) + 1)
    p_w = ((h_w_out[1] - 1) * stride[1] - h_w_in[1] + dilation[1] * (kernel_size[1] - 1) + 1)
    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (math.floor(p_w / 2), math.ceil(p_w / 2))


def truncated_normal(t, mean=0.0, std=0.01):
    """Return a normal draw truncated to +/-2 std (rejection-sampled). `t` seeds the
    shape/dtype/device; assign the return value -- out-of-bound resampling rebinds it."""
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
        if cond.any():
            t = torch.where(cond, torch.nn.init.normal_(torch.ones_like(t), mean=mean, std=std), t)
        else:
            break
    return t


class NN_SC_Weight_Clipper(object):
    """
    Clipper for NN weights/bias into the stochastic-computing range: 'norm' rescales to
    the full range on the first call, then 'clip' clamps on subsequent calls; both
    quantize to `bitwidth` bits. Apply via module.apply(clipper).
    """
    def __init__(self, frequency=1, mode="bipolar", method="clip", bitwidth=8):
        self.frequency = frequency
        self.mode = mode
        self.method = method
        self.scale = 2 ** bitwidth

    def __call__(self, module):
        self.method = "clip" if self.frequency > 1 else "norm"
        if hasattr(module, 'weight'):
            self.clipping(module.weight.data)
        if hasattr(module, 'bias') and module.bias is not None:
            self.clipping(module.bias.data)
        self.frequency = self.frequency + 1

    def clipping(self, w):
        if self.mode == "unipolar":
            if self.method == "norm":
                w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w)) \
                    .mul_(self.scale).round_().clamp_(0.0, self.scale).div_(self.scale)
            elif self.method == "clip":
                w.clamp_(0.0, 1.0).mul_(self.scale).round_().clamp_(0.0, self.scale).div_(self.scale)
            else:
                raise TypeError(f"unknown method '{self.method}' in NN_SC_Weight_Clipper, expected 'clip' or 'norm'")
        elif self.mode == "bipolar":
            if self.method == "norm":
                w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w)).mul_(2).sub_(1) \
                    .mul_(self.scale / 2).round_().clamp_(-self.scale / 2, self.scale / 2).div_(self.scale / 2)
            elif self.method == "clip":
                w.clamp_(-1.0, 1.0).mul_(self.scale / 2).round_().clamp_(-self.scale / 2, self.scale / 2).div_(self.scale / 2)
            else:
                raise TypeError(f"unknown method '{self.method}' in NN_SC_Weight_Clipper, expected 'clip' or 'norm'")
        else:
            raise TypeError(f"unknown mode '{self.mode}' in NN_SC_Weight_Clipper, expected 'unipolar' or 'bipolar'")


def check_name(config: dict):
    """
    Check if name is available.
    """
    name = config.get('name', None)
    if name is not None:
        assert isinstance(name, str), logger.error(f'Invalid name: <{name}>; name should be a string.')
        name = name.lower()
    return name
