from parallel import draw_attractors_in_parallel, draw_single_attractor
from colors import create_linear_colormap, randomCyclicColormap
from lyapunov_exponents import compute_attractor_pixel_rate, createAttractors
from equations import Equation
from config import RenderConfig, CreateConfig, BASE_DIR

import os; import glob

base_dir = BASE_DIR / "config_changes2"
create_cfg = CreateConfig(Equation('trig'), base_dir=base_dir)
render_config = RenderConfig(base_dir=base_dir)

def main():

    # TODO parallelize this too
    # find EXAMPLES random attractors with CREATE_ITERATIONS iters for each one
    createAttractors(create_cfg=create_cfg)

    input_files = glob.glob(f"{create_cfg.out_path}/*.json")
    cmaps = ["binary", "gray", "twilight", "hot", "magma"]
    #cmaps = randomCyclicColormap(5)

    # TODO move this to the inside of the rendering
    # hard-filter out unimportant attractors
    files = []
    for file in input_files:
        js = os.path.splitext(file)[0] + ".png"
        if os.path.exists(js) and compute_attractor_pixel_rate(js) > 0.02:
            files.append(file)

    print(f'{len(input_files)=}, {len(files)=}')

    assert len(input_files) > len(files) > 0
    print("Filtered: ", f'{len(input_files)=}, {len(files)=}')

    os.makedirs(render_config.dir, exist_ok=True) # TODO move this out of main
    print(f"Rendering directory: {render_config.dir}")

    for cmap_name in cmaps:
        print('#'*20, end='\n')
        print(f"Rendering with colormap {cmap_name}...")
        print('#'*20, end='\n\n')

        # rendering multiple attractors is a perfect task for multiprocessing! cpu go brr
        results = draw_attractors_in_parallel(
            files,
            render_config,
            cmap=create_linear_colormap(preset=cmap_name),
            n_processes=4, # leave some CPU for other tasks
            batch_size=10,
        )


if __name__ == "__main__":
    main() # using main so that multiprocessing issues are avoided on mac
