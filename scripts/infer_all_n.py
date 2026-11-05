"""Generate cohort videos at N=128,192,256 and full length; no metrics."""

from evaluation.cli import run


if __name__ == "__main__":
    raise SystemExit(run(["--mode", "infer-all-n", *__import__("sys").argv[1:]]))
