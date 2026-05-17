"""
Generates oo_engrams_prose.jsonl — same 5 OO concepts,
written as natural language Q&A instead of raw JSON blobs.
250 copies of each = 1250 samples total.
"""
import json

SAMPLES = [
    {
        "prompt": "Explain the 5 Organic Laws of the Operating Organism D+ Policy Engine and how they affect module actions.",
        "answer": (
            "The D+ Policy Engine enforces 5 hardcoded Organic Laws that govern every persistent action in the Operating Organism:\n\n"
            "Law 0 — Common Good: If benefit < 0.10, the action is QUARANTINED. All actions must serve the system.\n"
            "Law 1 — Non-Harm: If harm > 0.70, the action is FORBIDDEN outright.\n"
            "Law 2 — Transparency: If the reason field is NULL, the action is FORBIDDEN. All actions must be justified.\n"
            "Law 3 — Reversibility: If reversibility < 0.40, the engine returns COMPENSATE — the action is allowed but must have a rollback plan.\n"
            "Law 4 — Dignity: Self-modifying code with harm > 0.30 is QUARANTINED to protect system integrity.\n\n"
            "Every bare-metal action MUST be evaluated by calling `oo_organic_eval(&org, &action, ctx)` before execution. "
            "Bypassing D+ breaks the sovereign contract and is strictly forbidden."
        ),
    },
    {
        "prompt": "Implement a module that allocates a buffer for KV Cache in the bare-metal environment.",
        "answer": (
            "In the bare-metal llm-baremetal environment, standard `malloc()` is forbidden. "
            "All memory must be allocated through the OO-RAM zone system using `oo_ram_alloc(zone, size)`.\n\n"
            "The SSM Engine KV Cache must be placed in the WARM zone (read-write, address 0x500000):\n\n"
            "```c\n"
            "void* allocate_kv_cache(size_t size) {\n"
            "    return oo_ram_alloc(OO_ZONE_WARM, size);\n"
            "}\n"
            "```\n\n"
            "Key rules:\n"
            "- Use OO_ZONE_COLD for read-only model weights.\n"
            "- Use OO_ZONE_WARM for read-write runtime buffers like KV cache.\n"
            "- Never write to FROZEN or SENTINEL zones — this triggers the Warden.\n"
            "- Never use `malloc()` — it causes a compilation failure in freestanding mode."
        ),
    },
    {
        "prompt": "Integrate a new inference engine using the official OO Mamba Bridge interface.",
        "answer": (
            "To act as the cognitive core of the Operating Organism, an inference engine must implement "
            "the functions defined in `oo_mamba_bridge.h`:\n\n"
            "- `oo_engine_init()` — Initialize the engine on startup.\n"
            "- `oo_engine_generate(prompt, out, max_tokens)` — Run inference. Must respect D+ and thermal context.\n"
            "- `oo_engine_embed(text, vec)` — Produce embedding vectors for memory and routing.\n"
            "- `oo_engine_set_speed(float factor)` — Thermal governor hook. Must scale down matrix ops when thermal limits are hit.\n\n"
            "Example implementation:\n\n"
            "```c\n"
            "#include \"oo_mamba_bridge.h\"\n\n"
            "int oo_engine_generate(const char *prompt, char *out, int max_tokens) {\n"
            "    return mamba_generate_internal(prompt, out, max_tokens);\n"
            "}\n"
            "```\n\n"
            "Ensure `oo_engine_set_speed` actually reduces compute intensity when the thermal governor signals overload."
        ),
    },
    {
        "prompt": "Save a new memory state to disk using the bare-metal NeuralFS.",
        "answer": (
            "In the Operating Organism, direct disk writes are governed. "
            "All memory persistence must go through NeuralFS, which automatically invokes D+ before committing to physical storage.\n\n"
            "Use `oo_neuralfs_write(&nfs, key, content, tags, tag_count)` to save state:\n\n"
            "```c\n"
            "int save_memory_state(const char* key, const char* content) {\n"
            "    const char* tags[] = {\"memory\", \"state\"};\n"
            "    return oo_neuralfs_write(&nfs, key, content, tags, 2);\n"
            "}\n"
            "```\n\n"
            "NeuralFS handles D+ evaluation internally before every write. "
            "Never write directly to disk — all persistence must route through NeuralFS."
        ),
    },
    {
        "prompt": "List the commands used to evaluate and apply the Halt Policy in the OO Runtime REPL.",
        "answer": (
            "The OO Runtime REPL provides the following commands for the SSM Halt Policy:\n\n"
            "- `/mind_halt_probe` — Checks for active loops in the inference engine.\n"
            "- `/mind_halt_policy [threshold] [on|off]` — Configures the halt policy threshold and enables or disables it.\n"
            "- `/mind_halt_policy_save` — Persists the current halt policy configuration to `repl.cfg`.\n"
            "- `/mind_halt_policy_diff` — Shows the delta between the runtime policy and the on-disk `repl.cfg` version.\n"
            "- `/mind_audit` — Returns a global health report of all active sidecars and halting hooks.\n\n"
            "Use `/mind_audit` regularly to verify the halting system is active during long reasoning loops."
        ),
    },
]

OUTPUT_PATH = "/hdd_data/mamba_training_data/oo_engrams_prose.jsonl"
COPIES = 250

with open(OUTPUT_PATH, "w") as f:
    for sample in SAMPLES:
        for _ in range(COPIES):
            f.write(json.dumps(sample) + "\n")

total = len(SAMPLES) * COPIES
print(f"Written {total} samples ({len(SAMPLES)} concepts × {COPIES} copies) to {OUTPUT_PATH}")
