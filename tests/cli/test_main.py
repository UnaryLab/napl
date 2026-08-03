import contextlib
import io

from napl.main import main, parse_commandline_args


def test_help_describes_cli_modes_and_flags():
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            parse_commandline_args(['-h'])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError('argparse help should exit')

    help_text = output.getvalue()
    for expected in (
        '-sim',
        'simulate a NAPL program',
        '-user',
        'convert a user Python program',
        '-target',
        'accuracy',
        'latency',
        'energy',
        '-out',
        '.generated/',
        '-syn',
        'AST-linking',
    ):
        assert expected in help_text


def test_parse_requested_forms_and_defaults():
    args = parse_commandline_args(['-sim', 'program.napl'])
    assert args.sim is True
    assert args.syn is False
    assert args.user is False
    assert args.program == 'program.napl'
    assert args.target == 'accuracy'
    assert args.out == '.generated/'

    args = parse_commandline_args([
        '-sim', '-user', 'program.py', '-target=latency', '-out', 'generated',
    ])
    assert args.sim is True
    assert args.user is True
    assert args.program == 'program.py'
    assert args.target == 'latency'
    assert args.out == 'generated'

    args = parse_commandline_args(['-syn', 'program.napl'])
    assert args.sim is False
    assert args.syn is True
    assert args.program == 'program.napl'


def test_execution_paths_report_not_implemented_and_fail():
    for argv in (
        ['-sim', 'program.napl'],
        ['-sim', '-user', 'program.py'],
        ['-syn', 'program.napl'],
        ['-sim'],
        ['-syn'],
        ['-user', 'program.py'],
        [],
    ):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exit_code = main(argv)
        assert exit_code != 0
        assert 'not implemented' in output.getvalue().lower()


if __name__ == '__main__':
    test_help_describes_cli_modes_and_flags()
    test_parse_requested_forms_and_defaults()
    test_execution_paths_report_not_implemented_and_fail()
    print('Test passed.')
