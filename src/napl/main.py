import argparse
import sys


def parse_commandline_args(argv=None):
    """Parse the NAPL command-line argument surface."""
    parser = argparse.ArgumentParser(
        description='NAPL command-line interface.',
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        '-sim',
        action='store_true',
        help='simulate a NAPL program.',
    )
    modes.add_argument(
        '-syn',
        action='store_true',
        help='generate hardware by AST-linking NAPL simulation classes to implementation RTL.',
    )
    parser.add_argument(
        '-user',
        action='store_true',
        help='convert a user Python program to a NAPL program, then simulate it; use with -sim.',
    )
    parser.add_argument(
        '-target',
        choices=('accuracy', 'latency', 'energy'),
        default='accuracy',
        help='optimization objective for -sim -user (default: accuracy).',
    )
    parser.add_argument(
        '-out',
        default='.generated/',
        metavar='PATH',
        help='generated-program path for -sim -user (default: .generated/).',
    )
    parser.add_argument(
        'program',
        nargs='?',
        metavar='PROGRAM',
        help='NAPL or user Python program path.',
    )
    parser.add_argument(
        '-r', '--run_dir',
        default='tests/test_outputs',
        help='legacy run directory; execution remains not implemented.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_commandline_args(argv)

    if args.user and not args.sim:
        print('NAPL user-program conversion is not implemented; use -sim -user PROGRAM.', file=sys.stderr)
        return 1

    if args.sim:
        if args.program is None:
            print('NAPL program simulation is not implemented; provide PROGRAM after -sim.', file=sys.stderr)
            return 1
        if args.user:
            print('NAPL user-program conversion and simulation are not implemented.', file=sys.stderr)
        else:
            print('NAPL program simulation is not implemented.', file=sys.stderr)
        return 1

    if args.syn:
        if args.program is None:
            print('NAPL hardware generation is not implemented; provide PROGRAM after -syn.', file=sys.stderr)
            return 1
        print('NAPL hardware generation by AST-linking simulation classes to implementation RTL is not implemented.', file=sys.stderr)
        return 1

    print('NAPL CLI execution is not implemented; use -sim or -syn.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
