"""
Purpose: Call Ganak through Python. Note that these are all helper functions I got from
         ChatGPT, so we should test them thoroughly.

Date created: 2025-04-11
"""

import subprocess
import tempfile
import re

def write_cnf_to_tempfile(cnf_str):
    temp = tempfile.NamedTemporaryFile(mode='w+', suffix='.cnf', delete=False)
    temp.write(cnf_str)
    temp.flush()
    return temp.name

def run_ganak_on_cnf_file(cnf_filename, mode = 2, ganak_path='./ganak'):
    """
        Purpose: Call Ganak from Python.

        Inputs:
            - cnf_filename (string): The location of the CNF file.
            - mode (integer): Value of 2 corresponds to a complex weighted model count.
            - ganak_path (string): Invocation for Ganak.
        Output:
            - result: The terminal output.
    """
    result = subprocess.run(
        [ganak_path, "--mode={}".format(mode), cnf_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout, result.stderr

def parse_ganak_complex_output(output):
    """
    Purpose: Parse Ganak's output for the weighted model count (complex) and runtime.

    Inputs:
        - output: Ganak stdout as a string.

    Returns:
        A tuple: (complex model count, runtime in seconds)
    """
    complex_pattern = r"c s exact arb cpx ([+-]?\d+\.\d+e[+-]?\d+) \+ ([+-]?\d+\.\d+e[+-]?\d+)i"
    time_pattern = r"c o Total time \[Arjun\+GANAK\]: ([\d.]+)"

    complex_match = re.search(complex_pattern, output)
    time_match = re.search(time_pattern, output)

    count = None
    runtime = None

    if complex_match:
        real = float(complex_match.group(1))
        imag = float(complex_match.group(2))
        count = real + imag * 1j

    if time_match:
        runtime = float(time_match.group(1))

    return count, runtime