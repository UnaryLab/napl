import os
import subprocess
import sys


root_dir = os.path.abspath(os.path.dirname(__file__))
log_file = root_dir + '/sweep_test.log'


def sweep_test():
    failed = []
    with open(log_file, 'w') as f:
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.startswith('test_') and filename.endswith('.py'):
                    full_path = os.path.abspath(os.path.join(dirpath, filename))
                    print(f'Running: {full_path}')
                    result = subprocess.run(['python', full_path], stderr=f)
                    if result.returncode != 0:
                        failed.append(full_path)
    return failed


if __name__ == '__main__':
    failures = sweep_test()
    if failures:
        print(f'{len(failures)} test file(s) failed (see {log_file}):')
        for path in failures:
            print(f'  {path}')
    sys.exit(1 if failures else 0)
