from parallel import draw_attractors_in_parallel, draw_single_attractor
from colors import create_linear_colormap, randomCyclicColormap
from lyapunov_exponents import createAttractors
from equations import BuiltinEquation
from config import RenderConfig, CreateConfig, BASE_DIR

import os; import glob

base_dir = BASE_DIR / "config_changes2"
create_cfg = CreateConfig(
    BuiltinEquation('trig'),
    base_dir=base_dir,
    examples=50,
    iterations=100_000,
    )
render_config = RenderConfig(
    base_dir=base_dir,
    iterations=50_000_000,
    )

def main():

    # TODO parallelize this too
    # find EXAMPLES random attractors with CREATE_ITERATIONS iters for each one
    createAttractors(create_cfg=create_cfg)

    input_files = glob.glob(f"{create_cfg.out_path}/*.json")
    cmaps = ["binary", "gray", "twilight"] #, "hot", "magma"]
    #cmaps.extend(randomCyclicColormap(5))

    for cmap_name in cmaps:
        print('#'*20, end='\n'); print(f"Rendering with colormap {cmap_name}..."); print('#'*20, end='\n\n')

        # rendering multiple attractors is a perfect task for multiprocessing! cpu go brr
        results = draw_attractors_in_parallel(
            input_files,
            render_config,
            cmap=create_linear_colormap(preset=cmap_name),
            n_processes=4, # leave some CPU for other tasks
            batch_size=10,
        )

if __name__ == "__main__":
    main() # using main so that multiprocessing issues are avoided on mac
