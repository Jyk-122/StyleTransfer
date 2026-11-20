"""Independent content/style sizing shared by the project training and inference entries."""
from __future__ import annotations

import math

from PIL import Image


def add_reference_size_arguments(parser):
    for role in ("content", "style"):
        for dimension in ("height", "width", "max_pixels"):
            name = f"{role}_{dimension}"
            help_text = (f"{role.title()} reference {dimension}; height/width must be paired multiples of 16."
                         if dimension != "max_pixels" else
                         f"{role.title()} reference pixel limit; keep aspect ratio, downscale only. "
                         "Use instead of explicit height/width.")
            parser.add_argument("--" + name.replace("_", "-"), "--" + name,
                                dest=name, type=int, default=None, help=help_text)
    return parser


def validate_height_width(height, width, label):
    if (height is None) != (width is None):
        raise ValueError(f"{label}: specify both height and width")
    if height is not None and (min(height, width) < 16 or height % 16 or width % 16):
        raise ValueError(f"{label}: height and width must be positive multiples of 16")


def reference_size_settings(args):
    settings = {}
    for role in ("content", "style"):
        spec = {key: getattr(args, f"{role}_{key}", None) for key in ("height", "width", "max_pixels")}
        validate_height_width(spec["height"], spec["width"], role)
        if spec["max_pixels"] is not None:
            if spec["height"] is not None:
                raise ValueError(f"{role}: use either height/width or max_pixels, not both")
            if spec["max_pixels"] < 256:
                raise ValueError(f"{role}: max_pixels must be at least 256")
        settings[role] = spec
    return settings if any(value is not None for spec in settings.values() for value in spec.values()) else None


def reference_image_size(image, spec):
    """Return (width, height); unspecified references retain the pipeline's 1024²-area rule."""
    if spec["height"] is not None:
        return spec["width"], spec["height"]
    width, height = image.size
    if spec["max_pixels"] is None:
        # Match Flux2Unit_EditImageEmbedder, including upscaling and 32-pixel alignment.
        ratio = width / height
        width = math.sqrt(1024 * 1024 * ratio)
        height = width / ratio
        width, height = round(width / 32) * 32, round(height / 32) * 32
    else:
        scale = min(1.0, math.sqrt(spec["max_pixels"] / (width * height)))
        width, height = int(width * scale) // 16 * 16, int(height * scale) // 16 * 16
    if min(width, height) < 16:
        raise ValueError("Reference would be smaller than 16 pixels on one side; increase its pixel limit "
                         "or specify explicit height/width")
    return width, height


def prepare_reference_images(images, settings):
    if settings is None:
        return images  # Keep the original pipeline's preprocessing when no new flags are used.
    if not isinstance(images, (list, tuple)) or len(images) != 2:
        raise ValueError("Reference size options require edit_image=[content, style]")
    prepared = []
    for role, image in zip(("content", "style"), images):
        width, height = reference_image_size(image, settings[role])
        # Match the pipeline's bilinear resize + center crop, without stretching the image.
        scale = max(width / image.width, height / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
        left, top = round((image.width - width) / 2), round((image.height - height) / 2)
        prepared.append(image.crop((left, top, left + width, top + height)))
    return prepared


class ReferenceImageLoader:
    """Picklable dataset operator: load raw references without the target image's resize."""

    def __init__(self, base_path, settings):
        from diffsynth.core.data.operators import LoadImage, ToAbsolutePath
        self.load_image = ToAbsolutePath(base_path) >> LoadImage()
        self.settings = settings

    def __call__(self, paths):
        if not isinstance(paths, (list, tuple)) or len(paths) != 2:
            raise ValueError("Reference size options require edit_image=[content, style]")
        return prepare_reference_images([self.load_image(path) for path in paths], self.settings)
