import argparse

import pyfiglet

from napl.utils import bcolors


def parse_commandline_args():
    """
    parse command line inputs
    """
    parser = argparse.ArgumentParser(
        description='A neuro-adaptive programming language for general-purpose neuromorphic computing.')
    parser.add_argument('-r', '--run_dir', type=str, default='tests/test_outputs',
                        help = 'Run directory.')

    return parser.parse_args()


def main():
    args = parse_commandline_args()

    # set up banner
    ascii_banner = pyfiglet.figlet_format('NAPL')
    print(bcolors.Magenta + ascii_banner + bcolors.ENDC)
    ascii_banner = pyfiglet.figlet_format('UnaryLab')
    print(bcolors.Yellow + ascii_banner + bcolors.ENDC)
    ascii_banner = pyfiglet.figlet_format('https://github.com/UnaryLab/napl', font='term')
    print(bcolors.UNDERLINE + bcolors.Green + ascii_banner + bcolors.ENDC)


if __name__ == '__main__':
    main()

    