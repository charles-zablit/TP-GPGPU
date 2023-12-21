import subprocess
import pathlib
import logging
import re
import csv
from typing import List, Tuple, Optional
import time


class Program:
    root: pathlib.Path
    bin_name: str
    test_name: str
    use_nvprof: bool
    iters = 5
    run_time: Optional[float]

    def __init__(
        self, root: pathlib.Path, bin_name: str, test_name: str, use_nvprof: bool
    ) -> None:
        self.root = root
        self.bin_name = bin_name
        self.test_name = test_name
        self.use_nvprof = use_nvprof
        self.run_time = None

    @property
    def _bin_path(self) -> pathlib.Path:
        return pathlib.Path("./bin").joinpath(self.bin_name)

    @property
    def _results_dir(self) -> pathlib.Path:
        return self.root.parent.joinpath(f"results/{self.test_name}")

    def _log_time_estimate(self, iters: bool):
        if self.run_time is None:
            return
        if iters:
            run_time = self.iters * self.run_time
        else:
            run_time = self.run_time
        run_time = time.strftime("%Mm:%Ss", time.gmtime(run_time))
        logging.info(f"Will be done in ~{run_time}")

    def _results_to_csv(self, results: List[List[Tuple[float, float]]]):
        with open(self._results_dir.joinpath("measurements.csv"), "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["run", "epoch", "time", "acc"])
            for i, res in enumerate(results):
                for epoch_i, (time_, acc) in enumerate(res):
                    writer.writerow([i, epoch_i, time_, acc])

    def _make(self, param=None):
        cmd = ["make"]
        if param is None:
            logging.info("Building")
        else:
            cmd.append(param)
            logging.info(f"Building with param {param}")
        res = subprocess.run(
            cmd,
            cwd=self.root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if res.returncode != 0:
            logging.error(f"Error cleaning {res.stderr}")
            return
        logging.info("Done building")

    def _make_results_dir(self):
        logging.info(f"Creating results dir at '{str(self._results_dir)}'")
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def profile_gprof(self):
        self._make("clean")

        logging.info("Compiling in profile mode")
        self._make("profile")

        start = time.time()
        logging.info("Running program")
        self._log_time_estimate(False)
        res = subprocess.run(
            [f"./{str(self._bin_path)}"],
            cwd=self.root.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if res.returncode != 0:
            logging.error(f"Error profiling {res.stderr}")
            return
        self.run_time = time.time() - start

        logging.info("Running gprof")
        gprof_out_path = self._results_dir.joinpath("gprof.out")
        res = subprocess.run(
            f"gprof -Q -b ./{str(self._bin_path)} > '{str(gprof_out_path)}'",
            cwd=self.root.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
        )
        if res.returncode != 0:
            logging.error(f"Error running gprof {res}")
            return

        logging.info("Done profiling")

    def profile_nvprof(self):
        self._make("clean")

        logging.info("Compiling in profile mode")
        self._make("profile")

        nvprof_out_path = self._results_dir.joinpath("nvprof.out")
        logging.info("Running nvprof")
        self._log_time_estimate(False)
        res = subprocess.run(
            f"nvprof ./{str(self._bin_path)} > /dev/null",
            cwd=self.root.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
        )
        if res.returncode != 0:
            logging.error(f"Error running nvprof {res}")
            return

        with open(nvprof_out_path, "w") as f:
            f.write(res.stdout.decode("utf-8"))

        logging.info("Done with nvprof")

    def measure(self):
        self._make("clean")

        logging.info("Compiling")
        self._make()

        results: List[List[Tuple[float, float]]] = []

        logging.info("Running measurements")
        self._log_time_estimate(True)
        for i in range(self.iters):
            logging.info(f"Measuring [{i+1}/{self.iters}]")
            results.append([])
            res = subprocess.run(
                [f"./{str(self._bin_path)}"],
                cwd=self.root.parent,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if res.returncode != 0:
                logging.error(f"Error measuring {res.stderr}")
                return
            logging.info("Done measuring")

            logging.info(f"Parsing output")
            out = res.stdout.decode("utf-8")
            matches_0 = re.findall(
                r"\[INFO (\d+\.\d+)\] starting accuracy (\d+\.\d+)", out
            )
            results[-1].append((0.0, float(matches_0[0][1])))
            matches_1 = re.findall(
                r"\[INFO (\d+\.\d+)\] start learning epoch (\d+)", out
            )
            matches_2 = re.findall(
                r"\[INFO (\d+\.\d+)\] epoch (\d+) accuracy (\d+\.\d+)", out
            )
            for match_1, match_2 in zip(matches_1, matches_2):
                epoch_1 = int(match_1[1])
                epoch_2 = int(match_2[1])
                assert epoch_1 == epoch_2
                start_time = float(match_1[0])
                end_time = float(match_2[0])
                acc = float(match_2[2])
                results[-1].append((end_time - start_time, acc))
            logging.info("Done parsing")
        logging.info("Done measuring")

        logging.info("Writing output")
        self._results_to_csv(results)
        logging.info("Done writing output")

    def run(self):
        self._make_results_dir()
        self.profile_gprof()
        if self.use_nvprof:
            self.profile_nvprof()
        self.measure()


def main():
    program = Program(
        pathlib.Path("/home/charles/Developer/TP-GPGPU/par"),
        "ann_par",
        "par_cublas",
        True,
    )
    program.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    logging.info("Running metrics")
    main()
    logging.info("Done running metrics")
