import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional


class Data:
    name: str
    escaped_name: str
    measurements_df: pd.DataFrame
    nvprof_api_df: Optional[pd.DataFrame]
    nvprof_activities_df: Optional[pd.DataFrame]
    gprof_df: pd.DataFrame
    
    order = [
        "sequential",
        "basic",
        "gpu_minibatch",
        "coalescing",
        "shared_transpose",
        "shared_gemm",
        "float16",
        "cublas",
    ]

    def __init__(self, path: pathlib.Path) -> None:
        name = path.name.replace("par_", "")
        self.escaped_name = name
        self.measurements_df = pd.read_csv(path.joinpath("measurements.csv"), header=0)
        gprof = defaultdict(list)
        for line in open(path.joinpath("gprof.out"), "r").readlines():
            match_ = re.search(r"^\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d*)\s*(\d+.\d+)\s*(\d+.\d+)\s*(.*)$", line)
            if match_ is None:
                match_ = re.search(r"^\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*(.*)$", line)
                if match_ is not None:
                    gprof["time"].append(float(match_.group(1)))
                    gprof["seconds cumulative"].append(float(match_.group(2)))
                    gprof["seconds self"].append(float(match_.group(3)))
                    gprof["name"].append(match_.group(4))
                    gprof["calls"].append(0.0)
                    gprof["ms/call self"].append(0.0)
                    gprof["ms/call total"].append(0.0)
            else:
                gprof["time"].append(float(match_.group(1)))
                gprof["seconds cumulative"].append(float(match_.group(2)))
                gprof["seconds self"].append(float(match_.group(3)))
                gprof["calls"].append(float( match_.group(4)))
                gprof["ms/call self"].append(float(match_.group(5)))
                gprof["ms/call total"].append(float(match_.group(6)))
                gprof["name"].append(match_.group(7))
        self.gprof_df = pd.DataFrame(gprof)

        if not path.joinpath("nvprof.out").exists():
            self.nvprof_activities_df = None
            self.nvprof_api_df = None
            return

        nvprof_activities = defaultdict(list)
        nvprof_api = defaultdict(list)
        api_mode = False
        for line in open(path.joinpath("nvprof.out"), "r").readlines():
            match_ = re.search(r"(\d+.\d+)%\s*(\d+.\d+\w+)\s*(\d+)\s*(\d+.\d+\w+)\s*(\d+.\d+\w+)\s*(\d+.\d+\w+)\s*(.*)$", line)
            if re.search(r"^\s*API calls:", line) is not None:
                api_mode = True
                self.nvprof_activities_df = pd.DataFrame(nvprof_activities)
            if match_ is None:
                continue
            if api_mode:
                nvprof_api["Time(%)"].append(float(match_.group(1)))
                nvprof_api["Time"].append(parse_to_seconds(match_.group(2)))
                nvprof_api["Calls"].append(float(match_.group(3)))
                nvprof_api["Avg"].append(parse_to_seconds(match_.group(4)))
                nvprof_api["Min"].append(parse_to_seconds(match_.group(5)))
                nvprof_api["Max"].append(parse_to_seconds(match_.group(6)))
                nvprof_api["Name"].append(match_.group(7))
            else:
                nvprof_activities["Time(%)"].append(float(match_.group(1)))
                nvprof_activities["Time"].append(parse_to_seconds(match_.group(2)))
                nvprof_activities["Calls"].append(float(match_.group(3)))
                nvprof_activities["Avg"].append(parse_to_seconds(match_.group(4)))
                nvprof_activities["Min"].append(parse_to_seconds(match_.group(5)))
                nvprof_activities["Max"].append(parse_to_seconds(match_.group(6)))
                nvprof_activities["Name"].append(match_.group(7))
        self.nvprof_api_df = pd.DataFrame(nvprof_api)

    @property
    def plot_name(self) -> str:
        return "\n".join(self.escaped_name.split("_")).title()

    @property
    def name(self) -> str:
        return " ".join(self.escaped_name.split("_")).title()

def parse_to_seconds(time_str: str) -> float:
    time_match = re.search(r"(\d+.\d+)(\w+)", time_str)
    time_ = float(time_match.group(1))
    if time_match.group(2) == "s":
        return time_
    if time_match.group(2) == "ms":
        return time_ * 1e-3
    if time_match.group(2) == "us":
        return time_ * 1e-6
    if time_match.group(2) == "ns":
        return time_ * 1e-9
    raise Exception("No match")

def main():
    sns.set_theme(rc={"figure.dpi": 200, "figure.figsize": (10, 8)})

    root = pathlib.Path("/home/charles/Developer/TP-GPGPU/results")
    out_path = root.parent.joinpath("plots")
    out_path.mkdir(exist_ok=True)

    results: Dict[str, List[Data]] = dict()
    for dir_name in os.listdir(root):
        results[dir_name] = sorted(
            [Data(dir_) for dir_ in root.joinpath(dir_name).iterdir()],
            key=lambda x: Data.order.index(x.escaped_name),
        )
    base_line = results[os.listdir(root)[0]][0]

    plot_gprof(out_path, results)
    plot_nvprof(out_path, results)
    plot_accuracy_over_time(out_path, results)
    plot_epoch_time(out_path, results)
    plot_speedup(out_path, results, base_line)

def plot_gprof(out_path, results: Dict[str, List[Data]]):
    logging.info("Plotting GProf")
    for gpu, gpu_results in results.items():
        for res in gpu_results:
            if res.gprof_df.empty:
                continue
            data = list(res.gprof_df["time"][:5])
            data.append(100-sum(data))
            labels= list(res.gprof_df["name"][:5])
            labels = list(map(lambda x: x.split("(")[0], labels))
            labels.append("others")
            plt.pie(data, labels = labels, autopct='%.0f%%')
            plt.title(f"(GProf) 5 most time consuming functions\n{res.name} ({gpu})")
            path = out_path.joinpath(f"gprof_{res.escaped_name}_{gpu}")
            plt.savefig(f"{str(path)}")
            plt.clf()

def plot_nvprof(out_path, results: Dict[str, List[Data]]):
    logging.info("Plotting NVProf")
    for gpu, gpu_results in results.items():
        for res in gpu_results:
            if res.nvprof_api_df is not None and not res.nvprof_api_df.empty:
                data = list(res.nvprof_api_df["Time(%)"][:5])
                labels= list(res.nvprof_api_df["Name"][:5])
                labels = list(map(lambda x: x.split("(")[0], labels))
                if 100-sum(data) > 1:
                    data.append(100-sum(data))
                    labels.append("others")
                plt.pie(data, labels = labels, autopct='%.0f%%')
                plt.title(f"(NVProf API) 5 most time consuming functions\n{res.name} ({gpu})")
                path = out_path.joinpath(f"nvprofapi_{res.escaped_name}_{gpu}")
                plt.savefig(f"{str(path)}")
                plt.clf()

            if res.nvprof_activities_df is not None and not res.nvprof_activities_df.empty:
                data = list(res.nvprof_activities_df["Time(%)"][:5])
                labels= list(res.nvprof_activities_df["Name"][:5])
                labels = list(map(lambda x: x.split("(")[0], labels))
                if 100-sum(data) > 1:
                    data.append(100-sum(data))
                    labels.append("others")
                plt.pie(data, labels = labels, autopct='%.0f%%')
                plt.title(f"(NVProf Activities) 5 most time consuming functions\n{res.name} ({gpu})")
                path = out_path.joinpath(f"nvprofactivities_{res.escaped_name}_{gpu}")
                plt.savefig(f"{str(path)}")
                plt.clf()

def plot_speedup(out_path, results, base_line):
    logging.info("Plotting speedup")
    df_baseline = base_line.measurements_df.groupby(["run"])["time"].mean()
    df_dict = defaultdict(list)
    for gpu, gpu_results in results.items():
        for res in gpu_results[1:]:
            for i, time_ in enumerate(res.measurements_df.groupby(["run"])["time"].mean()):
                df_dict["time"].append(df_baseline.iloc[i] / time_)
                df_dict["optimization"].append(res.plot_name)
                df_dict["GPU"].append(gpu)
    df = pd.DataFrame(df_dict)
    barplot = sns.barplot(data=df, x="time", y="optimization", hue="GPU", orient="h")
    for i in barplot.containers:
        barplot.bar_label(i, fmt="%.3f", padding=10.0)
    barplot.set_xlim(0, None)
    barplot.set_xlabel("Average epoch ratio to baseline")
    barplot.set_title("Speedup ratio (5 runs)")
    fig = barplot.get_figure()
    path = out_path.joinpath(f"speedup")
    fig.savefig(f"{str(path)}")
    plt.clf()

def plot_epoch_time(out_path, results):
    logging.info("Plotting average time per epoch")
    df_dict = defaultdict(list)
    for gpu, gpu_results in results.items():
        for res in gpu_results:
            for time_ in res.measurements_df.groupby(["run"])["time"].mean():
                df_dict["time"].append(time_)
                df_dict["optimization"].append(res.plot_name)
                df_dict["GPU"].append(gpu)
    df = pd.DataFrame(df_dict)
    barplot = sns.barplot(data=df, x="time", y="optimization", hue="GPU", orient="h")
    for i in barplot.containers:
        barplot.bar_label(i, fmt="%.3f", padding=10.0)
    barplot.set_xlim(0, None)
    barplot.set_xlabel("Time (s)")
    barplot.set_title("Average time per epoch (5 runs)")
    fig = barplot.get_figure()
    path = out_path.joinpath(f"avg-epoch-time")
    fig.savefig(f"{str(path)}")
    plt.clf()

def plot_accuracy_over_time(out_path, results):
    logging.info("Plotting accuracy/epoch")
    for gpu, gpu_results in results.items():
        for res in gpu_results:
            lineplot = sns.lineplot(data=res.measurements_df, x="epoch", y="acc", hue="run")
            lineplot.set_ylim(0, 100)
            lineplot.set_xlim(-1, 41)
            lineplot.set_xlabel("Epoch")
            lineplot.set_ylabel("Accuracy (%)")
            lineplot.set_title(f"Accuracy over time (5 runs)\n{res.name} ({gpu})")
            fig = lineplot.get_figure()
            path = out_path.joinpath(f"acc-over-time_{res.escaped_name}_{gpu}")
            fig.savefig(f"{str(path)}")
            plt.clf()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    logging.info("Running metrics")
    main()
    logging.info("Done running metrics")
