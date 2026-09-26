"""Build the optional MIT-licensed compact SQLite postings extension."""
import argparse
import platform
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/runtime/erpostings.dylib"))
    parser.add_argument("--sqlite-include", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src/blocking/pack_postings.c"
    flags = ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
    include = args.sqlite_include
    if include is None and platform.system() == "Darwin":
        include = next((p for p in (Path("/opt/homebrew/opt/sqlite/include"),
            Path("/usr/local/opt/sqlite/include")) if (p / "sqlite3ext.h").exists()), None)
        if include is None:
            raise SystemExit("Pass --sqlite-include for loadable-extension SQLite headers; Apple SDK headers disable them.")
    if include:
        flags += ["-I", str(include)]
    subprocess.run(["cc", "-O3", "-Wall", "-Wextra", *flags, str(source), "-o", str(args.output)], check=True)
    print(args.output.resolve())

if __name__ == "__main__": main()
