"""Start the Blender MCP addon from a checked-out addon.py file.

Run this with a non-background Blender process, usually under Xvfb on a
headless remote server:

    xvfb-run -a blender --python scripts/blender_mcp_bootstrap.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_addon(addon_path: Path):
    spec = importlib.util.spec_from_file_location("dataevolver_blender_mcp_addon", addon_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Blender MCP addon from {addon_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    addon_env = os.environ.get("BLENDER_MCP_ADDON", "").strip()
    if addon_env:
        addon_path = Path(addon_env).expanduser()
    else:
        addon_path = Path(__file__).resolve().parent / "addon.py"
    if not addon_path.exists():
        raise FileNotFoundError(f"BLENDER_MCP_ADDON does not exist: {addon_path}")

    port = int(os.environ.get("BLENDER_MCP_PORT", "9876"))
    addon = _load_addon(addon_path)
    addon.register()

    # The addon creates bpy.types.blendermcp_server during register().
    import bpy  # type: ignore

    bpy.context.scene.blendermcp_port = port
    server = getattr(bpy.types, "blendermcp_server", None)
    if server is not None and getattr(server, "port", port) != port:
        server.stop()
        bpy.types.blendermcp_server = addon.BlenderMCPServer(port=port)
        bpy.types.blendermcp_server.start()

    print(f"DataEvolver Blender MCP bootstrap ready on 127.0.0.1:{port}")


main()
