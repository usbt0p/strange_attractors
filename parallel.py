import os
import logging
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from config import RenderConfig
from lyapunov_exponents import loadAttractor, generateAttractorFromParameters, drawAttractor
from lyapunov_exponents import compute_attractor_pixel_rate

def draw_single_attractor(filename : str, render_cfg : RenderConfig, cmap):
    """Worker function to render a single attractor file
    from its parameters.
    This is designed to be as a worker function for multiprocessing, or
    can be used standalone."""

    if render_cfg.iterations is None:
        raise ValueError("render_iterations must be specified")

    try:
        logging.info(f"Processing {filename}")
        params = loadAttractor(filename)
        x, y, xmin, xmax, ymin, ymax = generateAttractorFromParameters(
            params, render_cfg.iterations)
        logging.info("Correctly loaded parameters.")

        name = os.path.basename(filename).split('.')[0]
        drawAttractor(
                name,
                xmin, xmax,
                ymin, ymax,
                x, y,
                render_cfg,
                cmap=cmap,
            )
        logging.info(f"Finished rendering {filename}")
        return f"Successfully processed {filename}"
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        raise e

def draw_attractors_in_parallel(
    filenames : list[str],
    render_cfg : RenderConfig,
    cmap,
    n_processes=cpu_count()-2,
    batch_size=20,
):
    """Render multiple attractor files in parallel using multiprocessing.
    Sets up a pool of worker processes and processes in batches to manage memory usage.
    """
    logging.info(f"Processing {len(filenames)} files using {n_processes} processes...")

    filenames = filter_attr_files(filenames, threshold=render_cfg.filter_threshold)

    os.makedirs(render_cfg.dir, exist_ok=True)
    print(f"Rendering directory: {render_cfg.dir}")

    try:
        # TODO make this less verbose
        for i in range(0, len(filenames), batch_size):
            print(f"Batch {i//batch_size + 1}: processing files {i+1} to {min(i + batch_size, len(filenames))}...")
            batch = filenames[i:i + batch_size] # this is mapped to the worker

            with Pool(processes=n_processes) as pool:
                # Use partial because the worker function needs extra positional args bersides keyword args
                worker_func = partial(draw_single_attractor, render_cfg=render_cfg, cmap=cmap)
                # Use imap for progress tracking
                results = list(tqdm(
                    # TODO change imap for map since its much faster, set smaller batch sizes
                    # and just use tqdm on the outer for loop
                    pool.imap(worker_func, batch),
                    total=len(batch),
                    desc="Rendering..."
                ))

            # results summary
            successful = sum(1 for r in results if r.startswith("Successfully"))
            failed = len(results) - successful
            print(f"\tBatch {i//batch_size + 1}: {successful} successful, {failed} failed")

    except KeyboardInterrupt: # allow graceful exit on Ctrl+C
        print("Process interrupted by user. Exiting...")

    return results

def filter_attr_files(in_filenames : list[str], threshold : float = 0.02):
    """Hard-filter out unimportant attractors by removing the ones
    with little 'pixel density'.
    """
    files = []
    for file in in_filenames:
        js = os.path.splitext(file)[0] + ".png"
        if os.path.exists(js) and compute_attractor_pixel_rate(js) > threshold:
            files.append(file)

    print(f'{len(in_filenames)=}, {len(files)=}')
    assert len(in_filenames) > len(files) > 0
    print("Filtered: ", f'{len(in_filenames)=}, {len(files)=}')

    return files
