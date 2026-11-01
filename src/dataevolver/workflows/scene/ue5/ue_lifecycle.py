"""Let PIE and SDL process pending teardown before requesting editor exit."""
import time
import unreal


def quit_when_idle(receipt, save):
    started = time.monotonic()
    idle_ticks = 0
    handle = None

    def tick(delta):
        nonlocal idle_ticks
        elapsed = time.monotonic() - started
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
        idle_ticks = idle_ticks + 1 if not world else 0
        if not (idle_ticks >= 3 and elapsed >= 2) and elapsed < 30:
            return
        unreal.unregister_slate_post_tick_callback(handle)
        ready = idle_ticks >= 3 and elapsed >= 2
        receipt['editor_shutdown'] = dict(pie_world_absent=not world, idle_ticks=idle_ticks,
                                         elapsed_s=elapsed, ready=ready,
                                         scope='Editor event drain; not proof of engine teardown correctness')
        if not ready:
            receipt['success'] = False
            receipt['exception'] = 'PIE did not finish within the shutdown deadline'
        save()
        unreal.SystemLibrary.quit_editor()

    handle = unreal.register_slate_post_tick_callback(tick)
