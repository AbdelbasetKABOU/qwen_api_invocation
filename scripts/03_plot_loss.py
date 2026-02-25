import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("./outputs/qwen2p5_apibench_lora")

def find_latest_trainer_state(root: Path) -> Path:
    # prefer checkpoint-* folders, pick the largest step number
    ckpts = []
    for p in root.glob("checkpoint-*"):
        try:
            step = int(p.name.split("-")[-1])
            ckpts.append((step, p))
        except ValueError:
            pass
    if ckpts:
        ckpt = sorted(ckpts, key=lambda x: x[0])[-1][1]
        state = ckpt / "trainer_state.json"
        if state.is_file():
            return state
    # fallback: search anywhere
    states = list(root.rglob("trainer_state.json"))
    if not states:
        raise FileNotFoundError(f"No trainer_state.json under {root}")
    return states[0]

def main():
    state_path = find_latest_trainer_state(ROOT)
    print("Using:", state_path)

    d = json.loads(state_path.read_text(encoding="utf-8"))
    logs = d.get("log_history", [])
    steps = [x["step"] for x in logs if "loss" in x and "step" in x]
    loss  = [x["loss"] for x in logs if "loss" in x and "step" in x]

    if not steps:
        print("No loss entries found in log_history.")
        return

    plt.figure()
    plt.plot(steps, loss)
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.title("Training loss vs step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss.png", dpi=150)    
    plt.show()

if __name__ == "__main__":
    main()

