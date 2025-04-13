from pathlib import Path

def get_project_root() -> Path:
    """Lấy thư mục gốc của dự án."""
    return Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    print("Project root path:", get_project_root())
