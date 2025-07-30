import os
import subprocess


# Change this to the root directory if needed
root_dir = os.path.abspath(os.path.dirname(__file__))
log_file = root_dir + '/sweep_test.log'


def sweep_test():
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith('test_') and filename.endswith('.py'):
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                print(f'Running: {full_path}')
                with open(log_file, 'w') as f:
                    subprocess.run(['python', full_path], stderr=f)
    

if __name__ == '__main__':
    sweep_test()

