"""Bundle OpenMP with a fallback rpath while sharing an installed runtime.

Stock LightGBM uses @rpath/libomp.dylib. A private @loader_path reference produced
by delocate loads a second runtime when stock is imported too, which can abort.
Keep the shared install name and search Homebrew/MacPorts before our bundled copy.
"""
import base64
import csv
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def repair(wheel, destination, archs):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repaired = root / "repaired"
        subprocess.run(["delocate-wheel", "--require-archs", archs, "-w", str(repaired), wheel], check=True)
        artifact, = repaired.glob("*.whl")
        unpacked = root / "unpacked"
        with zipfile.ZipFile(artifact) as archive:
            archive.extractall(unpacked)
        native = unpacked / "avenue_lightgbm/lib/lib_lightgbm.dylib"
        runtime = unpacked / "avenue_lightgbm/.dylibs/libomp.dylib"
        assert native.is_file() and runtime.is_file()
        dependencies = subprocess.check_output(["otool", "-L", str(native)], text=True)
        old, = [line.strip().split(" (", 1)[0] for line in dependencies.splitlines()
                if "libomp.dylib (" in line]
        subprocess.run(["install_name_tool", "-change", old, "@rpath/libomp.dylib", str(native)], check=True)
        subprocess.run(["install_name_tool", "-id", "@rpath/libomp.dylib", str(runtime)], check=True)
        for path in ("/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib",
                     "/opt/local/lib/libomp", "@loader_path/../.dylibs"):
            subprocess.run(["install_name_tool", "-add_rpath", path, str(native)], check=True)
        for path in (runtime, native):
            subprocess.run(["codesign", "--force", "--sign", "-", str(path)], check=True)
        record, = unpacked.glob("*.dist-info/RECORD")
        rows = []
        for path in sorted(unpacked.rglob("*")):
            if path.is_file() and path != record:
                data = path.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                rows.append([path.relative_to(unpacked).as_posix(), "sha256=" + digest, len(data)])
        rows.append([record.relative_to(unpacked).as_posix(), "", ""])
        content = io.StringIO(newline="")
        csv.writer(content).writerows(rows)
        record.write_text(content.getvalue(), encoding="utf-8")
        with zipfile.ZipFile(destination / artifact.name, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(unpacked.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(unpacked).as_posix())


if __name__ == "__main__":
    repair(*sys.argv[1:])
