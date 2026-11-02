"""Score the selected cohort at N=128 with image and flow metrics."""

from evaluation.cli import run


if __name__ == "__main__":
    raise SystemExit(run(["--mode", "metrics-128", *__import__("sys").argv[1:]]))
