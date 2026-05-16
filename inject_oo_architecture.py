import json
import uuid
from pathlib import Path

def create_oo_gold_sample(goal, worker, summary, artifacts=None, risks=None, recommendations=None):
    """Generates a synthetic reasoning trace matching the Qwen-distilled JSONL schema."""
    task_id = f"gold_oo_{uuid.uuid4().hex[:8]}"
    result = {
        "task_id": task_id,
        "worker": worker,
        "status": "completed",
        "summary": summary,
        "artifacts": artifacts or [],
        "risks": risks or [],
        "recommendations": recommendations or [],
        "needs_validation": False,
        "metadata": {"source": "oo_baremetal_gold_injection"}
    }
    return {
        "id": task_id,
        "type": "successful_task",
        "content": {
            "task": {"goal": goal, "id": task_id},
            "result": result
        }
    }

def main():
    premium_data_path = Path("/hdd_data/mamba_training_data/premium_reasoning.jsonl")
    if not premium_data_path.exists():
        print(f"Error: Could not find {premium_data_path}. Ensure it is mounted and copied.")
        return

    samples = []
    
    # --- Sample 1: The 5 Organic Laws ---
    samples.append(create_oo_gold_sample(
        goal="Explain the 5 Organic Laws of the Operating Organism D+ Policy Engine and how they affect module actions.",
        worker="architect",
        summary="The D+ Policy Engine evaluates all persistent actions using 5 hardcoded laws: Law 0 (Common Good, quarantines if <0.10), Law 1 (Non-harm, forbids if >0.70), Law 2 (Transparency, forbids if reason==NULL), Law 3 (Reversibility, compensates if <0.40), Law 4 (Dignity, quarantines self-modifying code if harm >0.30). Any bare-metal action MUST pass `oo_organic_eval(&org, &action, ctx)` before execution.",
        risks=["Bypassing D+ is strictly forbidden and breaks the sovereign contract."],
        recommendations=["Always initialize a DplusAction struct with accurate harm and benefit metrics before calling oo_organic_eval."]
    ))

    # --- Sample 2: Memory Zones & Allocation ---
    samples.append(create_oo_gold_sample(
        goal="Implement a module that allocates a buffer for KV Cache in the bare-metal environment.",
        worker="c_developer",
        summary="In llm-baremetal, standard malloc() is forbidden. The OO-RAM is strictly zoned. The SSM Engine KV cache must be allocated in the WARM zone (0x500000, Read-Write) using `oo_ram_alloc(OO_ZONE_WARM, size)`.",
        artifacts=[{
            "name": "kv_alloc.c",
            "type": "code",
            "content": "void* allocate_kv_cache(size_t size) {\n    return oo_ram_alloc(OO_ZONE_WARM, size);\n}"
        }],
        risks=["Using malloc() will cause a compilation failure in freestanding mode.", "Writing to FROZEN or SENTINEL zones will trigger the Warden."],
        recommendations=["Always use `oo_ram_alloc`.", "Use OO_ZONE_COLD for read-only model weights."]
    ))

    # --- Sample 3: The Mamba Bridge API ---
    samples.append(create_oo_gold_sample(
        goal="Integrate a new inference engine using the official OO Mamba Bridge interface.",
        worker="engine_developer",
        summary="To act as the cognitive core, the inference engine must implement the functions defined in `oo_mamba_bridge.h`. This includes `oo_engine_init()`, `oo_engine_generate()`, `oo_engine_embed()`, and the thermal governor hook `oo_engine_set_speed(float factor)`.",
        artifacts=[{
            "name": "mamba_bridge_impl.c",
            "type": "code",
            "content": "#include \"oo_mamba_bridge.h\"\n\nint oo_engine_generate(const char *prompt, char *out, int max_tokens) {\n    // Must be governed by thermal context and D+\n    return mamba_generate_internal(prompt, out, max_tokens);\n}"
        }],
        recommendations=["Ensure `oo_engine_set_speed` actually scales down matrix operations when thermal limits are hit."]
    ))

    # --- Sample 4: NeuralFS and Persistence ---
    samples.append(create_oo_gold_sample(
        goal="Save a new memory state to disk using the bare-metal NeuralFS.",
        worker="memory_developer",
        summary="Direct disk writes are governed. All memory writes must pass through NeuralFS via `oo_neuralfs_write(&nfs, key, content, tags, tag_count)`. NeuralFS automatically invokes D+ before committing the write to physical storage.",
        artifacts=[{
            "name": "save_memory.c",
            "type": "code",
            "content": "int save_memory_state(const char* key, const char* content) {\n    const char* tags[] = {\"memory\", \"state\"};\n    return oo_neuralfs_write(&nfs, key, content, tags, 2);\n}"
        }]
    ))

    # --- Sample 5: The Runtime REPL Commands ---
    samples.append(create_oo_gold_sample(
        goal="List the commands used to evaluate and apply the Halt Policy in the OO Runtime REPL.",
        worker="operator",
        summary="The runtime supports strict halting commands for the SSM: `/mind_halt_probe` checks loops, `/mind_halt_policy [threshold] [on|off]` configures it, `/mind_halt_policy_save` persists to repl.cfg, and `/mind_halt_policy_diff` shows deltas between runtime and disk.",
        recommendations=["Use `/mind_audit` to get a global health report of the active sidecars and halting hooks."]
    ))

    print(f"Opening {premium_data_path} to inject OO Gold Samples...")
    
    # Inject each sample 200 times to create a massive density of OO knowledge
    injection_count = 200
    total_injected = 0
    
    with premium_data_path.open("a", encoding="utf-8") as f:
        for s in samples:
            for _ in range(injection_count):
                f.write(json.dumps(s) + "\n")
                total_injected += 1
                
    print(f"[Success] Injected {total_injected} highly-duplicated OO Gold Samples into the premium dataset!")
    print("The Mamba 3 model will now natively learn the bare-metal architecture during pre-training.")

if __name__ == "__main__":
    main()
