"""Ideal-proxy mass policy extracted from the native synthesis backend."""
import math


def number(value, label, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected finite number: " + label)
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError("Out of range: " + label)
    return value


def mass_properties(instance):
    x, y, z = instance["dimensions_m"]
    shape = instance["collider"]["shape"]
    if shape == "SPHERE":
        if max(x, y, z) - min(x, y, z) > 1e-8:
            raise ValueError("A spherical proxy requires uniform dimensions")
        volume = math.pi * x**3 / 6
    elif shape == "BOX":
        volume = x * y * z
    else:
        raise ValueError("Native rigid adapter covers BOX and SPHERE; use a qualified external solver for other shapes")
    policy = instance["mass"]
    if policy.get("mode") == "fixed":
        mass = number(policy["kg"], "mass kg", 1e-9)
    elif policy.get("mode") == "density":
        mass = number(policy["kg_m3"], "density kg/m3", 1e-9) * volume
    else:
        raise ValueError("Declare mass mode fixed or density before changing dimensions")
    inertia = ([mass * x*x / 10] * 3 if shape == "SPHERE" else
               [mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12])
    return {"mass_kg": mass, "proxy_volume_m3": volume, "inertia_diagonal_kg_m2": inertia,
            "basis": "Declared ideal collision proxy; empirical calibration is separate"}
