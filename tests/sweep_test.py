import collections
import os
import re
import subprocess
import sys


root_dir = os.path.abspath(os.path.dirname(__file__))
log_file = root_dir + '/sweep_test.log'

#: Output marking a run in which part of the file did not execute. Test files say so in
#: their own text ('MPS tests skipped.', 'CHECK NOT RUN'); under pytest the reporter adds
#: 'SKIPPED' and the 'N skipped' summary line.
skip_re = re.compile(r'\bskip(?:s|ped|ping)?\b|CHECK NOT RUN', re.IGNORECASE)


def sweep_test():
    """Run every test file and return a (status, path) list, statuses PASS/SKIP/FAIL."""
    results = []
    with open(log_file, 'w') as f:
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for filename in sorted(filenames):
                if filename.startswith('test_') and filename.endswith('.py'):
                    full_path = os.path.abspath(os.path.join(dirpath, filename))
                    # A file's merged output is held whole in memory before it is logged.
                    run = subprocess.run(['python', '-u', full_path], stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True)
                    if run.returncode != 0:
                        status = 'FAIL'
                    elif skip_re.search(run.stdout):
                        status = 'SKIP'
                    else:
                        status = 'PASS'
                    tail = '' if run.stdout.endswith('\n') or not run.stdout else '\n'
                    f.write(f'===== {status} {full_path}\n{run.stdout}{tail}')
                    print(f'{status}  {full_path}', flush=True)
                    results.append((status, full_path))
    return results


if __name__ == '__main__':
    results = sweep_test()
    tally = collections.Counter(status for status, _ in results)
    not_passed = [(status, path) for status, path in results if status != 'PASS']
    print(f'{tally["PASS"]} passed, {tally["SKIP"]} skipped, {tally["FAIL"]} failed '
          f'(see {log_file}){":" if not_passed else ""}')
    for status, path in not_passed:
        print(f'  {status} {path}')
    sys.exit(1 if tally['FAIL'] else 0)
