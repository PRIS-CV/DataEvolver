import inspect

from dataevolver.workflows.stages import image_to_3d, sam_segment, t2i_generate


def test_agentic_stage_options_are_opt_in_and_preserve_legacy_defaults():
    t2i = inspect.signature(t2i_generate.generate_images).parameters
    segment_one = inspect.signature(sam_segment.segment_image_sam3).parameters
    segment_all = inspect.signature(sam_segment.segment_all).parameters
    reconstruct = inspect.signature(image_to_3d.generate_meshes).parameters

    assert t2i["negative_prompt"].default == t2i_generate.DEFAULT_NEGATIVE_PROMPT
    assert segment_one["allow_fallback"].default is True
    assert segment_all["allow_fallback"].default is True
    assert reconstruct["allow_shape_only_fallback"].default is True
