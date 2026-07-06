import numpy as np
from PIL import Image


class imageSuperNet:
    """Image super-resolution upsampler.

    Uses RealESRGAN when available; falls back to PIL bicubic upsampling otherwise.
    The PIL bicubic fallback produces acceptable quality for texture generation
    without requiring the RealESRGAN ckpt or basicsr dependency.
    """

    def __init__(self, config) -> None:
        self._use_realesrgan = False
        self.upsampler = None

        try:
            import os
            import sys
            import types

            # Stub missing torchvision.transforms.functional_tensor
            # (this module was removed in newer torchvision versions)
            if "torchvision.transforms.functional_tensor" not in sys.modules:
                try:
                    import torchvision.transforms.functional_tensor  # noqa
                except ImportError:
                    import torchvision.transforms.functional as _tvf
                    stub = types.ModuleType("torchvision.transforms.functional_tensor")
                    stub.rgb_to_grayscale = _tvf.rgb_to_grayscale
                    sys.modules["torchvision.transforms.functional_tensor"] = stub

            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            ckpt_path = config.realesrgan_ckpt_path
            if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 1024:
                raise FileNotFoundError(
                    f"RealESRGAN ckpt missing or empty: {ckpt_path}"
                )

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=ckpt_path,
                dni_weight=None,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True,
                gpu_id=None,
            )
            self._use_realesrgan = True
            print("[imageSuperNet] Using RealESRGAN x4 upsampler.")

        except Exception as e:
            print(
                f"[imageSuperNet] RealESRGAN not available ({e}), "
                "using PIL bicubic 4x upsampling fallback."
            )
            self._use_realesrgan = False

    def __call__(self, image: Image.Image) -> Image.Image:
        if self._use_realesrgan:
            output, _ = self.upsampler.enhance(np.array(image))
            return Image.fromarray(output)
        else:
            # PIL bicubic fallback: 4x upscale (same scale as RealESRGAN)
            w, h = image.size
            return image.resize((w * 4, h * 4), Image.BICUBIC)
