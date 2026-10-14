from dataclasses import dataclass, astuple
from pathlib import Path
from equations import Equation

BASE_DIR: Path = Path("imgs")

@dataclass
class RenderConfig:
    """Config class for the rendering process.
    """
    base_dir: Path = BASE_DIR
    dir: Path | None = None
    iterations: int = 10_000_000
    width: int = 1080
    height: int = 1080
    interpolation: str = "histogram"
    pad_size: float = 0.5
    filter_threshold: float = 0.02

    def __post_init__(self):
        if self.dir is None:
            self.dir = Path(self.base_dir) / "render"

    def keys(self):
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        return getattr(self, key)

@dataclass
class CreateConfig:
    """Config class for the attractor creation process.
    """
    equation: Equation
    base_dir: Path = BASE_DIR
    out_path: Path | None = None
    examples: int = 50
    iterations: int = 100_000

    def __post_init__(self):
        if self.out_path is None:
            self.out_path = Path(self.base_dir) / "out"

if __name__ == "__main__":
    r = RenderConfig()
    print(r)
    # print the unpacked version
    print(astuple(r))
    def aaa(base_dir, dir, iterations, width, height, interpolation, pad_size, filter_threshold):
        print(base_dir, dir, iterations, width, height, interpolation, pad_size, filter_threshold)
    aaa(**r)
    print(r.keys())
