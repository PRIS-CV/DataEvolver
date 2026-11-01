import numpy as np
import torch
from PIL import Image

from dataevolver.adapters.chat_worker import ImageOps_for_chat, numpy_vision_rotary, numpy_text_rotary


def test_cpu_rotary_matches_reference_float32_for_axial_positions():
    class Module:
        inv_freq = 1. / (10000. ** (torch.arange(0, 64, 2).float()/64))
        attention_scaling = 1.
        @staticmethod
        def recomposition_frequencies(freq):
            return torch.cat([freq[0], freq[1]], dim=-1)
    module=Module()
    for n in [49,196,784]:
        positions=torch.stack((torch.arange(n)//14,torch.arange(n)%14))
        freqs=positions[...,None].float()*module.inv_freq
        actual=numpy_vision_rotary(module,torch.zeros(1),positions)
        expected=(module.recomposition_frequencies(freqs.cos()),module.recomposition_frequencies(freqs.sin()))
        assert all(torch.allclose(a,b,rtol=1e-6,atol=2e-7) for a,b in zip(actual,expected))
        assert all(a.dtype==torch.float32 and a.device.type=='cpu' for a in actual)


def test_chat_image_grid_is_fixed_for_different_aspect_ratios():
    for size in [(80,100),(100,80),(64,64)]:
        source=Image.fromarray(np.zeros((size[1],size[0],3),dtype=np.uint8))
        image=ImageOps_for_chat(source,224)
        assert image.size==(224,224) and image.mode=='RGB'
        assert source.size==size


def test_cpu_text_rotary_matches_multiaxis_reference_and_preserves_dtype():
    class Module:
        rope_type='default'
        inv_freq=1./(10000.**(torch.arange(0,64,2).float()/64))
        attention_scaling=1.
        mrope_section=[11,11,10]
        def recomposition_frequencies(self,freq):
            result=freq[0]
            for dim,offset in enumerate((1,2),start=1):
                index=slice(offset,self.mrope_section[dim]*3,3)
                result[...,index]=freq[dim,...,index]
            return torch.cat((result,result),dim=-1)
    module=Module()
    for n in [1,196,512]:
        positions=torch.stack([torch.arange(n),torch.arange(n)//14,torch.arange(n)%14])[:,None,:]
        frequencies=positions[...,None].float()*module.inv_freq
        actual=numpy_text_rotary(module,torch.zeros(1),positions)
        expected=(module.recomposition_frequencies(frequencies.cos()),module.recomposition_frequencies(frequencies.sin()))
        assert all(torch.allclose(a,b,rtol=1e-6,atol=2e-7) for a,b in zip(actual,expected))
