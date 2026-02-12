from parallel import draw_attractors_in_parallel, draw_single_attractor
from colors import create_linear_colormap, randomCyclicColormap
from lyapunov_exponents import compute_attractor_pixel_rate, createAttractors

import os; from tqdm import tqdm; import glob

CREATE_ITERATIONS = 100_000
EXAMPLES = 100
RENDER_ITERATIONS = 200_000_000 # long processing but good rendering

base_dir = "imgs/cyclicTest"

render_kwargs = {
    "width": 1080,
    "height": 1080,
    "dir": f"{base_dir}/render",
    "interpolation": "histogram",
    "pad_size": 0.5,
}
#logging.basicConfig(level=logging.INFO)

# TODO parallelize this too
# find EXAMPLES random attractors with CREATE_ITERATIONS iters for each one
createAttractors(out_path=f"{base_dir}/out", examples=EXAMPLES, iterations=CREATE_ITERATIONS)
input_files = glob.glob(f"{base_dir}/out/*.json")

#cmaps = ["binary", "gray", "twilight", "twilight_shifted", "magma"]

cmaps = randomCyclicColormap(5)
cmaps.append('magma')

# hard-filter out unimportant attractors
files = []
for file in input_files:
    js = os.path.splitext(file)[0] + ".png"
    if os.path.exists(js) and compute_attractor_pixel_rate(js) > 0.02:
        files.append(file)

assert len(input_files) > len(files) > 0
print("Filtered: ", f'{len(input_files)=}, {len(files)=}')
    
for cmap_name in cmaps:
    print('#'*20, end='\n')
    print(f"Rendering with colormap {cmap_name}...")
    print('#'*20, end='\n\n')

    render_kwargs['cmap'] = create_linear_colormap(preset=cmap_name) 

    # rendering multiple attracctor is a perfect task for multiprocessing! cpu go brr
    results = draw_attractors_in_parallel(
        files, 
        RENDER_ITERATIONS, 
        n_processes=6, # leave some CPU for other tasks
        batch_size=12,
        **render_kwargs
    )