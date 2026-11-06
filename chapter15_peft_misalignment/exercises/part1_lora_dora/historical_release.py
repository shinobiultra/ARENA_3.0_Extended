"""Lazy historical release plumbing, never used by CPU practice or import."""
def run_gpu_test(max_vram_gb=24.0):
    from chapter15_peft_misalignment.exercises.part1_lora_dora_adapter_controls import solutions
    return solutions.run_gpu_test(max_vram_gb=max_vram_gb)

def run_full_experiment(max_vram_gb=24.0):
    from chapter15_peft_misalignment.exercises.part1_lora_dora_adapter_controls import solutions
    return solutions.run_full_experiment(max_vram_gb=max_vram_gb)
