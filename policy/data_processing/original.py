"""Load byte-identical historical algorithms, never their standalone CLIs."""
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
import sys

SNAPSHOT = Path(__file__).resolve().parent / "vendor"
HASHES = {
    "arx_episode_plan_ppu4_20260812.py": "31f49232d6d8f7f5f0732b6a07b42d4e253c88b59adbad2472cb41dd74767711",
    "convert_arx2_packing_phone_to_lerobot_v21_10hz_codex.py": "ce2e49d4713c648ed31b010d81192bdba4135d6710ef34df2ce1c55adbad9e9d",
    "build_arx_multitask_77_tactile_sidecar_20260812.py": "2b4ce2b33be8670ea6c6ce8575657c2185ab17c7bc42d5859669096af8c1386a",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def algorithms():
    for filename, expected in HASHES.items():
        if file_sha256(SNAPSHOT / filename) != expected:
            raise ValueError(f"Historical algorithm snapshot changed: {filename}")
    sys.path.insert(0, str(SNAPSHOT))
    names = [Path(filename).stem for filename in HASHES]
    modules = []
    for name in names:
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (SNAPSHOT / f"{name}.py").resolve():
            raise RuntimeError(f"Refusing a shadowed historical module: {name}")
        modules.append(module)
    if modules[1].lerobot_dataset_module.CODEBASE_VERSION != "v2.1":
        raise RuntimeError("This converter requires the pinned LeRobot v2.1 writer, not LeRobot v3")
    return tuple(modules)  # planner, writer, tactile
