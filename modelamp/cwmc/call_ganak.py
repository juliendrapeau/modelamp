"""
Purpose: Compute the model count of a CNF formula using the probabilistic exact model counter Ganak through python.

Date created: 2025-04-11
"""

import subprocess
import re


class GanakSolver:
    """
    Compute the model count of CNF formuale using the probabilistic exact model counter Ganak (https://github.com/meelgroup/ganak).

    Attributes
    ----------
    ganak_path: str
        Path to the Ganak executable.
    ganak_kwargs: dict
        Additional arguments for Ganak.
    """

    def __init__(self, ganak_path="./ganak", ganak_kwargs={"mode": 2}):

        self.ganak_path = ganak_path
        self.ganak_kwargs = ganak_kwargs

    def _run_ganak(self, cnf_filename: str):
        """
        Call Ganak from Python to solve the CNF formula using subprocess.

        Parameters
        ----------
        cnf_filename: str
            Path to the CNF file.

        Returns
        -------
        stdout: str
            Standard output from Ganak.
        stderr: str
            Standard error from Ganak.
        """

        def kwargs_to_args(kwargs):
            args = []
            for k, v in kwargs.items():
                args += [f"--{k}", str(v)]
            return args

        result = subprocess.run(
            [self.ganak_path] + kwargs_to_args(self.ganak_kwargs) + [cnf_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout, result.stderr

    def _parse_ganak_output(self, output: str):
        """
        Parse Ganak's output for the (complex) (weighted) model count and runtime.

        Parameters
        ----------
        output: str
            Ganak stdout as a string.

        Returns
        -------
        count: complex
            The model count.
        runtime: float
            The runtime in seconds.
        """

        # Why does it work? Does it work for other ganak modes?
        count_pattern = (
            r"c s exact arb cpx ([+-]?\d+\.\d+e[+-]?\d+) \+ ([+-]?\d+\.\d+e[+-]?\d+)i"
        )
        time_pattern = r"c o Total time \[Arjun\+GANAK\]: ([\d.]+)"

        count_match = re.search(count_pattern, output)
        time_match = re.search(time_pattern, output)

        count = None
        runtime = None

        if count_match:
            real = float(count_match.group(1))
            imag = float(count_match.group(2))
            count = real + imag * 1j
        else:
            raise ValueError("Model count not found in Ganak output.")

        if time_match:
            runtime = float(time_match.group(1))
        else:
            raise ValueError("Runtime not found in Ganak output.")

        return count, runtime

    def solve(self, cnf_file_path: str, verbose=False, output_dir=None):
        """
        Compute the model count of a CNF formula using Ganak.

        Parameters
        ----------
        cnf_file_path: str
            The path to the CNF file.
        verbose: bool
            If True, print Ganak's stdout and stderr.
        output_dir: str
            If provided, save Ganak's stdout and stderr to this directory.

        Returns
        -------
        count: complex
            The complex weighted model count.
        runtime: float
            The runtime in seconds.
        """

        stdout, stderr = self._run_ganak(cnf_file_path)
        count, runtime = self._parse_ganak_output(stdout)

        if verbose:
            print("Ganak stdout: ", stdout)
            print("Ganak stderr: ", stderr)

        if output_dir is not None:
            with open(f"{output_dir}/ganak_stdout.txt", "w") as f:
                f.write(stdout)
            with open(f"{output_dir}/ganak_stderr.txt", "w") as f:
                f.write(stderr)

        return count, runtime
