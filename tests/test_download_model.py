"""Small, offline checks for checkpoint assembly."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.download_model import assemble


class ModelAssemblyTests(unittest.TestCase):
    def test_assembles_and_verifies_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            chunks = [f"part-{index}".encode() for index in range(5)]
            parts = []
            for index, chunk in enumerate(chunks):
                name = f"diffusion_pytorch_model.bin.part-{index:03d}"
                (directory / name).write_bytes(chunk)
                parts.append({"name": name, "size": len(chunk),
                              "sha256": hashlib.sha256(chunk).hexdigest()})
            combined = b"".join(chunks)
            manifest = {"source_filename": "diffusion_pytorch_model.bin",
                        "source_size": len(combined),
                        "source_sha256": hashlib.sha256(combined).hexdigest(),
                        "parts": parts}
            output = assemble(directory, manifest)
            self.assertEqual(output.read_bytes(), combined)
            self.assertEqual(assemble(directory, manifest), output)

    def test_rejects_corrupt_combined_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            name = "diffusion_pytorch_model.bin.part-000"
            (directory / name).write_bytes(b"part")
            manifest = {"source_filename": "diffusion_pytorch_model.bin",
                        "source_size": 4, "source_sha256": "0" * 64,
                        "parts": [{"name": name}]}
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                assemble(directory, manifest)
            self.assertFalse((directory / manifest["source_filename"]).exists())


if __name__ == "__main__":
    unittest.main()
