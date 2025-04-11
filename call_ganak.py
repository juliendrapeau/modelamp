"""
Purpose: Call Ganak through Python.

Date created: 2025-04-11
"""

import subprocess
import tempfile

def write_cnf_to_tempfile(cnf_str):
    temp = tempfile.NamedTemporaryFile(mode='w+', suffix='.cnf', delete=False)
    temp.write(cnf_str)
    temp.flush()
    return temp.name

def run_ganak_on_cnf_file(cnf_filename, mode = 2, ganak_path='./ganak'):
    result = subprocess.run(
        [ganak_path, "--mode={}".format(mode), cnf_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout, result.stderr

def parse_ganak_output(output):
    for line in output.splitlines():
        if line.startswith('c s exact'):
            return float(line.split()[-1])
    return None

import re

def parse_ganak_complex_output(output):
    """
    Parses Ganak output for weighted model counts in complex form.

    Returns:
        A complex number if found, else None.
    """
    pattern = r"c s exact arb cpx ([+-]?\d+\.\d+e[+-]?\d+) \+ ([+-]?\d+\.\d+e[+-]?\d+)i"
    match = re.search(pattern, output)
    if match:
        real = float(match.group(1))
        imag = float(match.group(2))
        return real + imag * 1j
    return None
