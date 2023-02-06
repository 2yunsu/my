"""Collage-making class definitions.

Arnheim 3 - Collage
Piotr Mirowski, Dylan Banarse, Mateusz Malinowski, Yotam Doron, Oriol Vinyals,
Simon Osindero, Chrisantha Fernando
DeepMind, 2021-2022
Copyright 2021 DeepMind Technologies Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import pathlib
from . import training
from . import video_utils
from .collage_generator import PopulationCollage
import cv2
import numpy as np
from .patches import get_segmented_data
import torch
import yaml


class CollageMaker():
  """Makes a single collage image.

  A collage image (aka tile) may involve 3x3 parallel evaluations.
  """

  def __init__(
      self,
      prompts,
      segmented_data,
      background_image,
      clip_model,
      file_basename,
      device,
      config):
    """Create a single square collage image.

    Args:
      prompts: list of prompts. Optional compositional prompts plus a global one
      segmented_data: patches for the collage
      background_image: background image for the collage
      clip_model: CLIP model
      file_basename: string, name to use for the saved files
      device: CUDA device
      config: dictionary with the following fields.

    Config fields:
        compositional_image: bool, whether to use 3x3 CLIPs
        output_dir: string, directory to save working and final images
        video_steps: int, how many steps between video frames; 0 is never
        population_video: bool, create a video with members of the population
        use_normalized_clip: bool, colour-correct images for CLIP evaluation
        use_image_augmentations: bool, use image augmentations in evaluation
        optim_steps: int, training steps for the collage
        pop_size: int, size of population being evolved
        evolution_frequency: bool, how many steps between evolution evaluations
        initial_search_size: int, initial random search size (1 means no search)
    """
    self._prompts = prompts
    self._segmented_data = segmented_data
    self._background_image = background_image
    self._clip_model = clip_model
    self._file_basename = file_basename
    self._device = device
    self._config = config
    self._compositional_image = self._config["compositional_image"]
    self._output_dir = self._config["output_dir"]
    self._use_normalized_clip = self._config["use_normalized_clip"]
    self._use_image_augmentations = self._config["use_image_augmentations"]
    self._optim_steps = self._config["optim_steps"]
    self._pop_size = self._config["pop_size"]
    self._population_video = self._config["population_video"]
    self._use_evolution = self._config["pop_size"] > 1
    self._evolution_frequency = self._config["evolution_frequency"]
    self._initial_search_size = self._config["initial_search_size"]

    self._video_steps = self._config["video_steps"]
    self._video_writer = None
    self._population_video_writer = None
    if self._video_steps:
      self._video_writer = video_utils.VideoWriter(
          filename=f"{self._output_dir}/{self._file_basename}.mp4")
      if self._population_video:
        self._population_video_writer = video_utils.VideoWriter(
            filename=f"{self._output_dir}/{self._file_basename}_pop_sample.mp4")

    if self._compositional_image:
      if len(self._prompts) != 10:
        raise ValueError(
            "Missing compositional image prompts; found {len(self._prompts)}")
      print("Global prompt is", self._prompts[-1])
      print("Composition prompts", self._prompts)
    else:
      if len(self._prompts) != 1:
        raise ValueError(
            "Missing compositional image prompts; found {len(self._prompts)}")
      print("CLIP prompt", self._prompts[0])

    # Prompt to CLIP features.
    self._prompt_features = training.compute_text_features(
        self._prompts, self._clip_model, self._device)
    self._augmentations = training.augmentation_transforms(
        224,
        use_normalized_clip=self._use_normalized_clip,
        use_augmentation=self._use_image_augmentations)

    # Create population of collage generators.
    self._generator = PopulationCollage(
        config=self._config,
        device=self._device,
        is_high_res=False,
        pop_size=self._pop_size,
        segmented_data=self._segmented_data,
        background_image=self._background_image)

    self._optimizer = training.make_optimizer(self._generator,
                                              self._config["learning_rate"])
    self._step = 0
    self._losses_history = []
    self._losses_separated_history = []

  @property
  def generator(self):
    return self._generator

  @property
  def step(self):
    return self._step

  def initialise(self):
    """Initialise the collage from checkpoint or search over hyper-parameters."""

    # If we use a checkpoint.
    if len(self._config["init_checkpoint"]) > 0:
      self.load(self._config["init_checkpoint"])
      return

    # If we do an initial random search.
    if self._initial_search_size > 1:
      print("\nInitial random search over "
            f"{self._initial_search_size} individuals")
      for j in range(self._pop_size):
        generator_search = PopulationCollage(
            config=self._config,
            device=self._device,
            pop_size=self._initial_search_size,
            is_high_res=False,
            segmented_data=self._segmented_data,
            background_image=self._background_image)
        self._optimizer = training.make_optimizer(generator_search,
                                                  self._config["learning_rate"])

        num_steps_search = self._config["initial_search_num_steps"]
        if num_steps_search > 1:
          # Run several steps of gradient descent?
          for step_search in range(num_steps_search):
            losses, _, _ = self._train(
                step=step_search, last_step=False,
                generator=generator_search)
        else:
          # Or simply let initialise the parameters randomly.
          _, _, losses, _ = training.evaluation(
              t=0,
              clip_enc=self._clip_model,
              generator=generator_search,
              augment_trans=self._augmentations,
              text_features=self._prompt_features,
              prompts=self._prompts,
              config=self._config,
              device=self._device)
        print(f"Search {losses}")
        idx_best = np.argmin(losses)
        print(f"Choose {idx_best} with loss {losses[idx_best]}")
        self._generator.copy_from(generator_search, j, idx_best)
        del generator_search
      print("Initial random search done\n")

      self._optimizer = training.make_optimizer(self._generator,
                                                self._config["learning_rate"])

  def load(self, path_checkpoint):
    """Load an existing generator from state_dict stored in `path`."""
    print(f"\nLoading spatial and colour transforms from {path_checkpoint}...")
    state_dict = torch.load(path_checkpoint, map_location=self._device.type)
    this_state_dict = self._generator.state_dict()
    if state_dict.keys() != this_state_dict.keys():
      print(f"Current and loaded state_dict do not match")
    for key in this_state_dict:
      this_shape = this_state_dict[key].shape
      shape = state_dict[key].shape
      if this_shape != shape:
        print(f"state_dict[{key}] do not match: {this_shape} vs. {shape}")
        print(f"Abort loading from checkpoint.")
        return
    print(f"Checkpoint {path_checkpoint} restored.")
    self._generator.load_state_dict(state_dict)

  def _train(self, step, last_step, generator):
    losses, losses_separated, img_batch = training.step_optimization(
        t=step,
        clip_enc=self._clip_model,
        lr_scheduler=self._optimizer,
        generator=generator,
        augment_trans=self._augmentations,
        text_features=self._prompt_features,
        prompts=self._prompts,
        config=self._config,
        device=self._device,
        final_step=last_step)
    return losses, losses_separated, img_batch

  def loop(self):
    """Main optimisation/image generation loop. Can be interrupted."""
    if self._step == 0:
      print("\nStarting optimization of collage.")
    else:
      print(f"\nContinuing optimization of collage at step {self._step}.")
      if self._video_steps:
        print("Aborting video creation (does not work when interrupted).")
        self._video_steps = 0
        self._video_writer = None
        self._population_video_writer = None

    while self._step < self._optim_steps:
      last_step = self._step == (self._optim_steps - 1)
      losses, losses_separated, img_batch = self._train(
          step=self._step, last_step=last_step, generator=self._generator)
      self._add_video_frames(img_batch, losses)
      self._losses_history.append(losses)
      self._losses_separated_history.append(losses_separated)

      if (self._use_evolution and self._step
          and self._step % self._evolution_frequency == 0):
        training.population_evolution_step(
            self._generator, self._config, losses)
      self._step += 1

  def high_res_render(self,
                      segmented_data_high_res,
                      background_image_high_res,
                      gamma=1.0,
                      show=False,##True
                      save=True,
                      no_background=False):
    """Save and/or show a high res render using high-res patches."""
    generator_cpu = PopulationCollage(
        config=self._config,
        device="cpu",
        is_high_res=True,
        pop_size=1,
        segmented_data=segmented_data_high_res,
        background_image=background_image_high_res)
    idx_best = np.argmin(self._losses_history[-1])
    lowest_loss = self._losses_history[-1][idx_best]
    print(f"Lowest loss: {lowest_loss} @ index {idx_best}: ")
    generator_cpu.copy_from(self._generator, 0, idx_best)
    generator_cpu = generator_cpu.to("cpu")
    generator_cpu.tensors_to("cpu")

    params = {"gamma": gamma,
              "max_block_size_high_res": self._config.get(
                  "max_block_size_high_res")}
    if no_background:
      params["no_background"] = True
    with torch.no_grad():
      img_high_res = generator_cpu.forward_high_res(params)
    img = img_high_res.detach().cpu().numpy()[0]

    img = np.clip(img, 0.0, 1.0)
    if save or show:
      # Swap Red with Blue
      if img.shape[2] == 4:
        print("Image has alpha channel")
        img = img[..., [2, 1, 0, 3]]
      else:
        img = img[..., [2, 1, 0]]
      img = np.clip(img, 0.0, 1.0) * 255
    if save:
      if no_background:
        image_filename = f"{self._output_dir}/{self._file_basename}_no_bkgd.png"
      else:
        image_filename = f"{self._output_dir}/{self._file_basename}.png"
      cv2.imwrite(image_filename, img)
    if show:
      video_utils.cv2_imshow(img)

    img = img[:, :, :3]

    return img

  def finish(self):
    """Finish video writing and save all other data."""
    if self._losses_history:
      losses_filename = f"{self._output_dir}/{self._file_basename}_losses"
      training.plot_and_save_losses(self._losses_history,
                                    title=f"{self._file_basename} Losses",
                                    filename=losses_filename,
                                    show=self._config["gui"])
    if self._video_steps:
      self._video_writer.close()
    if self._population_video_writer:
      self._population_video_writer.close()
    metadata_filename = f"{self._output_dir}/{self._file_basename}.yaml"
    with open(metadata_filename, "w") as f:
      yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
    last_step = self._step
    last_loss = float(np.amin(self._losses_history[-1]))
    return (last_step, last_loss)

  def _add_video_frames(self, img_batch, losses):
    """Add images from numpy image batch to video writers.

    Args:
      img_batch: numpy array, batch of images (S,H,W,C)
      losses: numpy array, losses for each generator (S,N)
    """
    if self._video_steps and self._step % self._video_steps == 0:
      # Write image to video.
      best_img = img_batch[np.argmin(losses)]
      self._video_writer.add(cv2.resize(
          best_img, (best_img.shape[1] * 3, best_img.shape[0] * 3)))
      if self._population_video_writer:
        laid_out = video_utils.layout_img_batch(img_batch)
        self._population_video_writer.add(cv2.resize(
            laid_out, (laid_out.shape[1] * 2, laid_out.shape[0] * 2)))


class CollageTiler():
  """Creates a large collage by producing multiple overlapping collages."""

  def __init__(self,
               prompts,
               fixed_background_image,
               clip_model,
               device,
               config):
    """Create CollageTiler.

    Args:
      prompts: list of prompts for the collage maker
      fixed_background_image: highest res background image
      clip_model: CLIP model
      device: CUDA device
      config: dictionary with the following fields below:

    Config fields used:
        width: number of tiles wide
        height: number of tiles high
        background_use: how to use the background, e.g. per tile or whole image
        compositional_image: bool, compositional for multi-CLIP collage tiles
        high_res_multiplier: int, how much bigger is the final high-res image
        output_dir: directory for generated files
        torch_device: string, either cpu or cuda
    """
    self._prompts = prompts
    self._fixed_background_image = fixed_background_image
    self._clip_model = clip_model
    self._device = device
    self._config = config
    self._tiles_wide = config["tiles_wide"]
    self._tiles_high = config["tiles_high"]
    self._background_use = config["background_use"]
    self._compositional_image = config["compositional_image"]
    self._high_res_multiplier = config["high_res_multiplier"]
    self._output_dir = config["output_dir"]
    self._torch_device = config["torch_device"]

    pathlib.Path(self._output_dir).mkdir(parents=True, exist_ok=True)
    self._tile_basename = "tile_y{}_x{}{}"
    self._tile_width = 448 if self._compositional_image else 224
    self._tile_height = 448 if self._compositional_image else 224
    self._overlap = 1. / 3.

    # Size of bigger image
    self._width = int(((2 * self._tiles_wide + 1) * self._tile_width) / 3.)
    self._height = int(((2 * self._tiles_high + 1) * self._tile_height) / 3.)

    self._high_res_tile_width = self._tile_width * self._high_res_multiplier
    self._high_res_tile_height = self._tile_height * self._high_res_multiplier
    self._high_res_width = self._high_res_tile_width * self._tiles_wide
    self._high_res_height = self._high_res_tile_height * self._tiles_high

    self._print_info()
    self._x = 0
    self._y = 0
    self._collage_maker = None
    self._fixed_background = self._scale_fixed_background(high_res=True)

  def _print_info(self):
    """Print some debugging information."""

    print(f"Tiling {self._tiles_wide}x{self._tiles_high} collages")
    print("Optimisation:")
    print(f"Tile size: {self._tile_width}x{self._tile_height}")
    print(f"Global size: {self._width}x{self._height} (WxH)")
    print("High res:")
    print(
        f"Tile size: {self._high_res_tile_width}x{self._high_res_tile_height}")
    print(f"Global size: {self._high_res_width}x{self._high_res_height} (WxH)")
    for i, tile_prompts in enumerate(self._prompts):
      print(f"Tile {i} prompts: {tile_prompts}")

  def initialise(self):
    """Initialise the collage maker, optionally from a checkpoint or initial search."""

    if not self._collage_maker:
      # Create new collage maker with its unique background.
      print(f"\nNew collage creator for y{self._y}, x{self._x} with bg")
      tile_bg, self._tile_high_res_bg = self._get_tile_background()
      video_utils.show_and_save(tile_bg, self._config,
                                img_format="SCHW", stitch=False,
                                show=self._config["gui"])
      prompts_x_y = self._prompts[self._y * self._tiles_wide + self._x]
      segmented_data, self._segmented_data_high_res = (
          get_segmented_data(
              self._config, self._x + self._y * self._tiles_wide))
      self._collage_maker = CollageMaker(
          prompts=prompts_x_y,
          segmented_data=segmented_data,
          background_image=tile_bg,
          clip_model=self._clip_model,
          file_basename=self._tile_basename.format(self._y, self._x, ""),
          device=self._device,
          config=self._config)
    self._collage_maker.initialise()

  def load(self, path):
    """Load an existing CollageMaker generator from state_dict stored in `path`."""
    self._collage_maker.load(path)

  def loop(self):
    """Re-entrable loop to optmise collage."""

    res_training = {}
    while self._y < self._tiles_high:
      while self._x < self._tiles_wide:
        if not self._collage_maker:
          self.initialise()
        self._collage_maker.loop()
        collage_img = self._collage_maker.high_res_render(
            self._segmented_data_high_res,
            self._tile_high_res_bg,
            gamma=1.0,
            show=self._config["gui"],
            save=True)
        self._collage_maker.high_res_render(
            self._segmented_data_high_res,
            self._tile_high_res_bg,
            gamma=1.0,
            show=False,
            save=True,
            no_background=True)
        self._save_tile(collage_img / 255)

        (last_step, last_loss) = self._collage_maker.finish()
        res_training[f"tile_{self._y}_{self._x}_loss"] = last_loss
        res_training[f"tile_{self._y}_{self._x}_step"] = last_step
        del self._collage_maker
        self._collage_maker = None
        self._x += 1
      self._y += 1
      self._x = 0

    # Save results of all optimisations.
    res_filename = f"{self._output_dir}/results_training.yaml"
    with open(res_filename, "w") as f:
      yaml.dump(res_training, f, default_flow_style=False, allow_unicode=True)

    return collage_img  # SHWC

  def _save_tile(self, img):
    background_image_np = np.asarray(img)
    background_image_np = background_image_np[..., ::-1].copy()
    filename = self._tile_basename.format(self._y, self._x, ".npy")
    np.save(f"{self._output_dir}/{filename}", background_image_np)

  def _save_tile_arrays(self, all_arrays):
    filename = self._tile_basename.format(self._y, self._x, "_arrays.npy")
    np.save(f"{self._output_dir}/{filename}", all_arrays)

  def _scale_fixed_background(self, high_res=True):
    """Get correctly sized background image."""

    if self._fixed_background_image is None:
      return None
    multiplier = self._high_res_multiplier if high_res else 1
    if self._background_use == "Local":
      height = self._tile_height * multiplier
      width = self._tile_width * multiplier
    elif self._background_use == "Global":
      height = self._height * multiplier
      width = self._width * multiplier
    return cv2.resize(self._fixed_background_image.astype(float),
                      (width, height))

  def _get_tile_background(self):
    """Get the background for a particular tile.

    This involves getting bordering imagery from left, top left, above and top
    right, where appropriate.
    i.e. tile (1,1) shares overlap with (0,1), (0,2) and (1,0)
    (0,0), (0,1), (0,2), (0,3)
    (1,0), (1,1), (1,2), (1,3)
    (2,0), (2,1), (2,2), (2,3)
    Note that (0,0) is not needed as its contribution is already in (0,1)

    Returns:
      background_image: small background for optimisation
      background_image_high_res: high resolution background
    """
    if self._fixed_background is None:
      tile_border_bg = np.zeros((self._high_res_tile_height,
                                 self._high_res_tile_width, 3))
    else:
      if self._background_use == "Local":
        tile_border_bg = self._fixed_background.copy()
      else:  # Crop out section for this tile.
        orgin_y = self._y * (self._high_res_tile_height
                             - math.ceil(self._tile_height * self._overlap)
                             * self._high_res_multiplier)
        orgin_x = self._x * (self._high_res_tile_width
                             - math.ceil(self._tile_width * self._overlap)
                             * self._high_res_multiplier)
        tile_border_bg = self._fixed_background[
            orgin_y : orgin_y + self._high_res_tile_height,
            orgin_x : orgin_x + self._high_res_tile_width, :]
    tile_idx = dict()
    if self._x > 0:
      tile_idx["left"] = (self._y, self._x - 1)
    if self._y > 0:
      tile_idx["above"] = (self._y - 1, self._x)
      if self._x < self._tiles_wide - 1:  # Penultimate on the row
        tile_idx["above_right"] = (self._y - 1, self._x + 1)

    # Get and insert bodering tile content in this order.
    if "above" in tile_idx:
      self._copy_overlap(tile_border_bg, "above", tile_idx["above"])
    if "above_right" in tile_idx:
      self._copy_overlap(tile_border_bg, "above_right", tile_idx["above_right"])
    if "left" in tile_idx:
      self._copy_overlap(tile_border_bg, "left", tile_idx["left"])

    background_image = self._resize_image_for_torch(
        tile_border_bg, self._tile_height, self._tile_width)
    background_image_high_res = self._resize_image_for_torch(
        tile_border_bg,
        self._high_res_tile_height,
        self._high_res_tile_width).to("cpu")

    return background_image, background_image_high_res

  def _resize_image_for_torch(self, img, height, width):
    # Resize and permute to format used by Collage class (SCHW).
    img = torch.tensor(cv2.resize(img.astype(float), (width, height)))
    if self._torch_device == "cuda":
      img = img.cuda()
    return img.permute(2, 0, 1).to(torch.float32)

  def _copy_overlap(self, target, location, tile_idx):
    """Copy area from tile adjacent to target tile to target tile."""

    big_height = self._high_res_tile_height
    big_width = self._high_res_tile_width
    pixel_overlap = int(big_width * self._overlap)

    filename = self._tile_basename.format(tile_idx[0], tile_idx[1], ".npy")
    # print(f"Loading tile {filename})
    source = np.load(f"{self._output_dir}/{filename}")
    if location == "above":
      target[0 : pixel_overlap, 0 : big_width, :] = source[
          big_height - pixel_overlap : big_height, 0 : big_width, :]
    if location == "left":
      target[:, 0 : pixel_overlap, :] = source[
          :, big_width - pixel_overlap : big_width, :]
    elif location == "above_right":
      target[
          0 : pixel_overlap, big_width - pixel_overlap : big_width, :] = source[
              big_height - pixel_overlap : big_height, 0 : pixel_overlap, :]

  def assemble_tiles(self):
    """Stitch together the whole image from saved tiles."""

    big_height = self._high_res_tile_height
    big_width = self._high_res_tile_width
    full_height = int((big_height + 2 * big_height * self._tiles_high) / 3)
    full_width = int((big_width + 2 * big_width * self._tiles_wide) / 3)
    full_image = np.zeros((full_height, full_width, 3)).astype("float32")

    for y in range(self._tiles_high):
      for x in range(self._tiles_wide):
        filename = self._tile_basename.format(y, x, ".npy")
        tile = np.load(f"{self._output_dir}/{filename}")
        y_offset = int(big_height * y * 2 / 3)
        x_offset = int(big_width * x * 2 / 3)
        full_image[y_offset : y_offset + big_height,
                   x_offset : x_offset + big_width, :] = tile[:, :, :]
    filename = "final_tiled_image"
    print(f"Saving assembled tiles to {filename}")
    video_utils.show_and_save(
        full_image, self._config, img_format="SHWC", stitch=False,
        filename=filename, show=self._config["gui"])
