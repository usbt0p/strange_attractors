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
Generation and rendering are configured with two dataclasses from `config.py`:

- `CreateConfig(equation, base_dir="imgs", out_path=None, examples=50, iterations=100_000)`
- `RenderConfig(base_dir="imgs", dir=None, iterations=10_000_000, width=1080, height=1080, interpolation="histogram", pad_size=0.5, filter_threshold=0.02)`

If `out_path` / `dir` are not given, they default to `<base_dir>/out` and `<base_dir>/render`, so changing `base_dir` moves both.

Generate attractors and save their parameters:
```python
from config import CreateConfig
from equations import BuiltinEquation
from lyapunov_exponents import createAttractors

create_cfg = CreateConfig(BuiltinEquation("abs"), base_dir="new_attractors", examples=100, iterations=10_000)
createAttractors(create_cfg)

# this will create 100 attractors and save their parameters in new_attractors/out, with
# a small 500x500 image preview for each one.
```

You can easily specify equation types using the `BuiltinEquation` class, which currently supports three type of attractor equations; trigonometric (`"trig"`), absolute value (`"abs"`) and quadratic (`"quadratic"`).
```python
create_cfg = CreateConfig(
    # specify your equation here
    equation=BuiltinEquation("abs", coef_bounds=(-1.0, 1.0)),
    base_dir="new_attractors",
)
```

Render a specific attractor from its parameters:
```python
from config import RenderConfig
from parallel import draw_single_attractor
from colors import create_linear_colormap, randomTriadColors

render_cfg = RenderConfig(
    base_dir="new_attractors",
    iterations=20_000_000,
    width=1000,
    height=1000,
)

draw_single_attractor("new_attractors/out/0_0.json",
                      render_cfg=render_cfg,
                      cmap=create_linear_colormap(colors=list(randomTriadColors(100, 255))))
```

Render multiple attractors in parallel:
```python
import glob
from multiprocessing import cpu_count
from config import RenderConfig
from parallel import draw_attractors_in_parallel
from colors import create_linear_colormap

render_cfg = RenderConfig(base_dir="new_attractors", iterations=20_000_000) # big but good quality

draw_attractors_in_parallel(
    glob.glob("new_attractors/out/*.json"),
    render_cfg,
    # recolor_base makes the background color white
    cmap=create_linear_colormap(preset='viridis', recolor_base={'n':1, 'color':(255,255,255)}),
    n_processes=cpu_count(),
    batch_size=50, # set this lower if you have less memory
)
```

Whole pipeline example: create multiple attractors, and render them in parallel with different preset and custom colormaps.
```python
import glob
from multiprocessing import cpu_count
from config import CreateConfig, RenderConfig, BASE_DIR
from colors import create_linear_colormap
from equations import BuiltinEquation
from lyapunov_exponents import createAttractors
from parallel import draw_attractors_in_parallel

base_dir = BASE_DIR / "set_num"
create_cfg = CreateConfig(BuiltinEquation("quadratic"), base_dir=base_dir, examples=100)
render_cfg = RenderConfig(base_dir=base_dir, iterations=20_000_000, width=1000, height=1000)

def main():
    createAttractors(create_cfg)
    files = glob.glob(f"{create_cfg.out_path}/*.json")

    cmaps = ['plasma', 'magma', 'viridis', 'cividis',
             'turbo', 'summer', 'autumn', 'winter', 'copper']
    # define a recoloring scheme to make the background white
    recolor = {'n':1, 'color':(255,255,255), 'resample_size':512}

    for cmap_name in cmaps:
        print(f"Rendering with colormap {cmap_name}...")
        # rendering multiple attractors is a perfect task for multiprocessing! cpu go brr
        draw_attractors_in_parallel(
            files,
            render_cfg,
            cmap=create_linear_colormap(preset=cmap_name, recolor_base=recolor),
            n_processes=cpu_count() - 4, # leave some CPU for other tasks
            batch_size=50,
        )

    # custom colormap example
    draw_attractors_in_parallel(
        files,
        render_cfg,
        cmap=create_linear_colormap(colors=[(255, 0, 0), (0, 255, 0)], recolor_base=recolor),
        n_processes=cpu_count() - 4,
        batch_size=50,
    )

if __name__ == "__main__":
    main() # needed for multiprocessing on macOS
```

### Colour options
All renders take a matplotlib colormap through `cmap`, usually built with `create_linear_colormap` from `colors.py`. With testing, the best results seem to come both from some matplotlib presets (`magma`, `plasma`, `Greys`, `gray`, `bone`, `copper`, `twilight`, `hot`...) and the `randomCyclicColormap`'s, which are designed to start and end at the same colour.

```python
from colors import (create_linear_colormap, randomTriadColors,
                    randomComplementaryColors, randomColoursGoldenRatio, randomCyclicColormap)

# matplotlib preset
# https://matplotlib.org/stable/users/explain/colors/colormaps.html
cmap = create_linear_colormap(preset="magma")

# custom gradient from RGB (0-255) tuples, evenly spaced
cmap = create_linear_colormap(colors=[(0, 0, 0), (2, 24, 107), (255, 145, 0)])

# custom gradient with explicit positions (0 to 1, one per colour)
cmap = create_linear_colormap(colors=[(255, 255, 255), (2, 24, 107), (255, 145, 0), (255, 255, 255)],
                              positions=[0.0, 0.4, 0.8, 1.0])

# replace the lowest n colours, i.e. the background, with a fixed colour (white here)
cmap = create_linear_colormap(preset="viridis", recolor_base={'n': 1, 'color': (255, 255, 255)})

# random harmonious palettes
cmap = create_linear_colormap(colors=list(randomTriadColors(100, 255)))
cmap = create_linear_colormap(colors=list(randomComplementaryColors()))
cmap = create_linear_colormap(colors=randomColoursGoldenRatio(n=4))

# random cyclic colormaps (start and end on the same colour), returns a list
cmaps = randomCyclicColormap(5)
```
With no arguments `create_linear_colormap()` returns `magma`.

---

A video generation utility is also provided in `attractor_video.py`, which can create videos of the attractor points being drawn over time.
An example of its usage:

```python
from lyapunov_exponents import loadAttractor, generateAttractorFromParameters

    RENDER_ITERATIONS = 200_000_000

    attractors = [15, 21, 1, 25, 13, 34]
    colors = ['viridis', 'afmhot', "binary", "gray", "twilight",
              "turbo", "BuPu", "YlGn", "YlOrRd", "Blues", "Reds", "Greens",
              "seismic", "Accent"]

    for n in attractors:
        print(f"Processing attractor {n}...")

        # loading pulls info from the metadata JSON to save you from configuration
        params = loadAttractor(f"examples/out/{n}_0.json")
        x, y, xmin, xmax, ymin, ymax = \
            generateAttractorFromParameters(params, RENDER_ITERATIONS)

        print(f"Attractor {n} loaded and generated.")

        for color in colors:
            videoAttractor(
                f"{n}_{color}",
                xmin, xmax,
                ymin, ymax,
                x, y,
                video_duration=10.0,
                fps=14,
                video_width=800,
                video_height=800,
                color_bias=0.25,
                pad_size=20,
                dir="videos",
                cmap=color,
                video_end_pad=2.0,
                experiment_name=f"colors"
            )

        print(f"Attractor {n} video generated.")
        print()
```
Outputs look something like this (but not as gifs):

<img src="examples/15_twilight_800x800_0.gif" width="500"/>

Outputs are encoded with `mp4v`, but if you want a mobile-compatible format like `yuv420p`, try this:
```bash
ffmpeg -i "inputvideo.mp4" -c:v libx264 -pix_fmt yuv420p -c:a aac "outvideo_mobile.mp4"
```
or for a gif:
```bash
ffmpeg \
  -i video.mp4 \
  -r 15 \ # fps
  # this filter pipeline removes dithering-like artifacts of scaling
  -vf "scale=512:-1,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  out.gif
```
---

When generating with different equations, you'll notice big differences between some operations, specialy trig operations like `np.sin` or `np.cos`. Take this into account when doing big runs. You can test this with the following sample code:

```python
import time, numpy as np
from equations import BuiltinEquation
from lyapunov_exponents import compute_attractor_core

for name in ("abs", "trig", "quadratic"):
    eq = BuiltinEquation(name)
    a, b = eq.random_coefs()
    compute_attractor_core(eq.step, a, b, 0.1, 0.1, 100)  # warm-up / compile
    t = time.perf_counter()
    compute_attractor_core(eq.step, a, b, 0.1, 0.1, 200_000_000)
    print(name, time.perf_counter() - t)
```
Results:
```bash
abs 0.6626970419965801
trig 3.0719616249989485
quadratic 0.7880107910023071
```

## TODO Some ideas to try
- [x] separate plotting and point generation in the drawing code.
- [ ] refactor the generateFilename function to be used by all
- [ ] PRIORITY: in the createAttractors function, ensure the number created is one with FILTERED pixel density! no default shitty attractors

- optimize:
    - [x] density computation (numba vs numpy)
    - [ ] chunked version for allowing parallel gen with huge iter counts
    - [x] optimize sequential functions with numba jit

- equations:
    - [ ] add some new from attractors book (ttriadic, tetradic, quintic, cosine and sphere projected)
    - [ ] Add symbolic equations and save them as parameters. That way, automatic equation generation will be a possibility (with arbitrary params).
    - [ ] Change the quadratic equation to use `a**x` instead of `a*x*x`.
    - [ ] Make higher dimensional attractors and project them into a smaller space.
    - [ ] Make 3D attractors and plot them in 3D.

- lyapunov:
    - [ ] Test the Lyapunov normalization `lyapunov /= num_iters`.
    - [ ] Implement fractal dimension calculation (need a meethod that yields all lyapunov exponents, nut just the highest).

- Perform data analysis on a big generated set of attractors + their parameters:
    - [ ] Augment parameters and calculate stats.
    - [ ] Cluster by attractor features.
    - [ ] Find out the relation the non-interesting generated attractors have with pixel density / exponents.
    - [ ] clusterby visual similarity
    - [x] find out a filter to discard non interesting attractors (besides lyapunov)

- colors:
    -[x] TODO IMPLEMENT THIS !! https://nicoguaro.github.io/posts/cyclic_colormaps/
    - [ ] in the matplotlib colo0rmaps page theres info about how to create colormap categories, try that:
        https://matplotlib.org/stable/users/explain/colors/colormaps.html#miscellaneous

- [x] Test the correct loading with seed points: all attractors should work when loaded.
- [x] Make a video / gif of the points appearing with their respective colour.
- [x] add padding to sides of drawings (just blank space at the limits)
- [ ] add transparency to the histogram mode
- [ ] add `exists_ok=True` for ignoring existing files

## References!
https://paulbourke.net/fractals/lyapunov/

https://sprott.physics.wisc.edu/fractals/booktext/SABOOK.PDF

https://www.nathanselikoff.com/training/tutorial-strange-attractors-in-c-and-opengl

https://www3.fi.mdp.edu.ar/fc3/SisDin2009/Clase%202/nousado/CHAOSCOPE/help/en/manual/attractors.htm

http://devmag.org.za/2012/07/29/how-to-choose-colours-procedurally-algorithms/

https://nicoguaro.github.io/posts/cyclic_colormaps/
