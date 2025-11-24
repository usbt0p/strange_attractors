# Strange attractors
Automatic generation of attractors by using Lyapunov exponents to calculate promising chaothic attractors (and discarding point or infinity attractors),
based on the idea of [Paul Bourke](https://paulbourke.net/fractals/lyapunov/).

## Some examples

| | | |
|-----------|-----------|-----------|
| <img src="examples/0.png" width="400"/> | <img src="examples/7.png" width="400"/> | <img src="examples/21.png" width="400"/> |
| <img src="examples/24.png" width="400"/> | <img src="examples/29.png" width="400"/> | <img src="examples/33.png" width="400"/> |
| <img src="examples/59.png" width="400"/> | <img src="examples/143_0.png" width="400"/> | <img src="examples/75.png" width="400"/> |
| <img src="examples/79.png" width="400"/> | <img src="examples/100.png" width="400"/> | <img src="examples/125.png" width="400"/> |
| <img src="examples/135.png" width="400"/> | <img src="examples/136.png" width="400"/> | <img src="examples/147.png" width="400"/> |

## Code examples
Generate attractors and save their parameters:
```python
from lyapunov_exponents import createAttractor
createAttractor(out_path="new_attractors/out", examples=100, iterations=10_000)

# this will create 100 attractors and save their parameters in the specified folder, with
# a small 500x500 image preview for each one. 
```

Render a specific attractor from its parameters:
```python
from parallel import draw_single_attractor
from colors import create_linear_colormap, randomTriadColors

colors = [randomTriadColors(100, 255)]

render_kwargs = {
        "width": 1000,
        "height": 1000,
        "cmap": create_linear_colormap(colors),
        "dir": "new_attractors/renders",
        "interpolation": "histogram",
        "density_sigma": 0,
        "pad_size": 0.5
    }

draw_single_attractor("new_attractors/out/attractor_0.json", 
                       render_iterations=20_000_000,
                       **render_kwargs)
``` 

Render multiple attractors in parallel:
```python
from parallel import render_attractors_in_parallel
from multiprocessing import cpu_count
import glob
from colors import create_linear_colormap

RENDER_ITERS = 20_000_000 # this is a big amount but gives good quality renders
render_kwargs = {
        "width": 1000,
        "height": 1000,
        "cmap": create_linear_colormap(preset='viridis', 
            # this makes the background color white 
            recolor_base={'n':1, 'color':(255,255,255)}),
        "dir": "new_attractors/renders",
        "interpolation": "histogram",
        "pad_size": 0.5
    }

render_attractors_in_parallel(glob.glob("new_attractors/out/*.json"), 
                              RENDER_ITERS,
                              num_workers=cpu_count(),
                              batch_size=50, # set this lower if you have less memory
                              **render_kwargs)
```

Whole pipeline example: create multiple attractors, and render them in parallel with different preset and custom colormaps.
```python
from parallel import draw_attractors_in_parallel, draw_single_attractor
from colors import create_linear_colormap

CREATE_ITERATIONS = 100_000
EXAMPLES = 100
RENDER_ITERATIONS = 20_000_000 # long processing but good rendering
base_dir = "set_num"

render_kwargs = {
    "width": 1000,
    "height": 1000,
    "cmap": create_linear_colormap(preset='viridis'),
    "dir": f"{base_dir}/renders",
    "interpolation": "histogram",
    "density_sigma": 0,
    "pad_size": 0.5
}
#logging.basicConfig(level=logging.INFO)

# find EXAMPLES random attractors with CREATE_ITERATIONS iters for each one
createAttractor(out_path=f"{base_dir}/out", examples=EXAMPLES, iterations=CREATE_ITERATIONS)

cmaps = ['plasma', 'magma', 'viridis', 'cividis', 
            'turbo', 'summer', 'autumn', 'winter', 'copper']
# define a recoloring scheme to make the background white
recolor = {'n':1, 'color':(255,255,255), 'resample_size':512}

for cmap_name in cmaps:
    print(f"Rendering with colormap {cmap_name}...")
    render_kwargs['cmap'] = create_linear_colormap(preset=cmap_name, 
                                                    recolor_base=recolor)

    # rendering multiple attracctor is a perfect task for multiprocessing! cpu go brr
    results = draw_attractors_in_parallel(
        glob.glob(f"{base_dir}/out/*.json"), 
        RENDER_ITERATIONS, 
        n_processes=cpu_count() - 4, # leave some CPU for other tasks
        batch_size=50,
        **render_kwargs
    )

# custom colormap example
render_kwargs['cmap'] = create_linear_colormap(colors=[(255, 0, 0),(0, 255, 0)] ,
                                                recolor_base=recolor)
results = draw_attractors_in_parallel(
        glob.glob(f"{base_dir}/out/*.json"), 
        RENDER_ITERATIONS, 
        n_processes=cpu_count() - 4, # leave some CPU for other tasks
        batch_size=50,
        **render_kwargs
    )
```

## TODO Some ideas to try 
- [ ] separate plotting and point generation in the drawing code.
- [ ] refactor the generateFilename function to be used by all

- [ ] Add symbolic equations and save them as parameters. That way, automatic equation generation will be a possibility (with arbitrary params).
- [x] Test the correct loading with seed points: all attractors should work when loaded.
- [ ] Change the quadratic equation to use `a**x` instead of `a*x*x`.
- [ ] Test the Lyapunov normalization `lyapunov /= num_iters`.
- [ ] Implement fractal dimension calculation(need a meethod that yields all lyapunov exponents, nut just the highest).
- [ ] Perform data analysis on a big generated set of attractors + their parameters:
    - [ ] Augment parameters and calculate stats.
    - [ ] Cluster by attractor features.
    - [ ] Find out the relation the non-interesting generated attractors have with pixel density / exponents.
- [ ] Make higher dimensional attractors and project them into a smaller space.
- [ ] Make 3D attractors and plot them in 3D.
- [ ] Make a video / gif of the points appearing with their respective colour.
- [x] add padding to sides of drawings (just blank space at the limits)
- [ ] add transparency to the histogram mode
- [ ] add `exists_ok=True` for ignoring existing files

## References!
https://paulbourke.net/fractals/lyapunov/

https://sprott.physics.wisc.edu/fractals/booktext/SABOOK.PDF

https://www.nathanselikoff.com/training/tutorial-strange-attractors-in-c-and-opengl

https://www3.fi.mdp.edu.ar/fc3/SisDin2009/Clase%202/nousado/CHAOSCOPE/help/en/manual/attractors.htm

http://devmag.org.za/2012/07/29/how-to-choose-colours-procedurally-algorithms/
