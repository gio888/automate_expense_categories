import os
import zipfile
from pathlib import Path
from fnmatch import fnmatch

GITIGNORE_PATH = Path(".gitignore")
OUTPUT_ZIP = Path("project_snapshot.zip")


def load_gitignore_patterns():
    if not GITIGNORE_PATH.exists():
        print("No .gitignore file found.")
        return []

    patterns = []
    with open(GITIGNORE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def should_ignore(file_path, patterns):
    relative_path = str(file_path)
    for pattern in patterns:
        if fnmatch(relative_path, pattern) or fnmatch(file_path.name, pattern):
            return True
    return False


def zip_project(base_dir="."):
    base_path = Path(base_dir).resolve()
    patterns = load_gitignore_patterns()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(base_path)

                if should_ignore(rel_path, patterns):
                    continue

                zipf.write(file_path, rel_path)
    print(f"✅ Zipped project to {OUTPUT_ZIP}")


if __name__ == "__main__":
    zip_project()
