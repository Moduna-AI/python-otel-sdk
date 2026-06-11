"""Environment file helpers."""

from pathlib import Path


class EnvFileReader:
    """Read values from simple dotenv-style files."""

    def __init__(self, env_path: Path | str = ".env") -> None:
        """Initialize the reader with the target dotenv path."""
        self.env_path = Path(env_path)

    def get(self, key: str) -> str | None:
        """Return a value from the dotenv file if present."""
        if not self.env_path.exists():
            return None

        for raw_line in self.env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip().strip("\"'")

        return None
