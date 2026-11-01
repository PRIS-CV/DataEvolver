"""Measure relative upper-hemisphere irradiance of an RGBE Radiance panorama.

shortcut: this reads standard RGBE scanline RLE, not XYZE or flat RGBE files.
The result has relative units until the source has radiometric calibration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def measure(path):
    path = Path(path)
    with path.open('rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Missing Radiance header')
            if not line.strip():
                break
            header.append(line)
        assert b'FORMAT=32-bit_rle_rgbe\n' in header, 'Expected RGBE Radiance'
        match = re.fullmatch(rb'-Y (\d+) \+X (\d+)\s*', f.readline())
        assert match, 'Expected top-to-bottom latitude-longitude orientation'
        height, width = map(int, match.groups())
        assert 8 <= width <= 32767 and height > 0
        weighted = upper = all_pixels = 0.0
        for y in range(height):
            assert f.read(4) == bytes([2, 2, width >> 8, width & 255])
            channels = []
            for _ in range(4):
                row = bytearray()
                while len(row) < width:
                    raw = f.read(1)
                    assert raw
                    n = raw[0]
                    assert n, 'Invalid empty RLE packet'
                    if n > 128:
                        value = f.read(1)
                        assert value
                        row.extend(value * (n-128))
                    else:
                        value = f.read(n)
                        assert len(value) == n
                        row.extend(value)
                assert len(row) == width
                channels.append(row)
            row_sum = sum(math.ldexp(.2126*r + .7152*g + .0722*b, e-136) if e else 0
                          for r,g,b,e in zip(*channels))
            all_pixels += row_sum
            if y < height/2:
                theta = math.pi * (y+.5) / height
                weighted += row_sum * math.sin(theta) * math.cos(theta)
                upper += row_sum
        irradiance = weighted * 2 * math.pi**2 / (width*height)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),width=width,height=height,
                mean_linear_luminance=all_pixels/(width*height),upper_mean_luminance=upper/(width*height/2),
                horizontal_diffuse_irradiance_relative=irradiance,
                units='relative RGB luminance units; not measured cd/m2 or lux',physical_calibration=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    print(json.dumps(measure(args.path),indent=2))
