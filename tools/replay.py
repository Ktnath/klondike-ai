import argparse
import csv
import os
import time
from ast import literal_eval
from typing import List, Dict, Any


def load_episode(path: str) -> List[Dict[str, Any]]:
    """Load one episode CSV into a list of step dicts."""
    steps: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            row["step"] = int(row["step"])
            row["action"] = int(row["action"])
            row["reward"] = float(row["reward"])
            row["done"] = row["done"].lower() == "true"
            row["epsilon"] = float(row["epsilon"])
            row["cumulative_reward"] = float(row["cumulative_reward"])
            try:
                row["observation"] = literal_eval(row["observation"])
            except Exception:
                row["observation"] = row["observation"]
            steps.append(row)
    return steps


def summarize_obs(obs: Any, length: int = 5) -> str:
    """Return a short string summary for an observation array."""
    if isinstance(obs, list):
        parts = ", ".join(f"{x:.2f}" for x in obs[:length])
        if len(obs) > length:
            parts += ", ..."
        return f"[{parts}]"
    return str(obs)


def replay_console(steps: List[Dict[str, Any]], speed: float) -> None:
    """Replay episode step by step in the console."""
    for row in steps:
        line = (
            f"Step {row['step']} | action={row['action']} | reward={row['reward']:.2f} | "
            f"epsilon={row['epsilon']:.2f} | done={row['done']} | "
            f"cumulative={row['cumulative_reward']:.2f}"
        )
        print(line)
        print(f"observation: {summarize_obs(row['observation'])}")
        time.sleep(max(speed, 0))


def episode_to_html(steps: List[Dict[str, Any]], output: str, speed: float) -> None:
    """Export the episode steps to a simple HTML replay."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    total_reward = steps[-1]["cumulative_reward"] if steps else 0.0
    rows = "\n".join(
        (
            "<tr>"
            f"<td>{s['step']}</td><td>{s['action']}</td>"
            f"<td>{s['reward']:.2f}</td><td>{s['epsilon']:.2f}</td>"
            f"<td>{s['done']}</td><td>{s['cumulative_reward']:.2f}</td>"
            "</tr>"
        )
        for s in steps
    )

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Replay {os.path.basename(output)}</title>
<style>
 table {{border-collapse: collapse; width: 100%;}}
 th, td {{border: 1px solid #ccc; padding: 4px; text-align: center;}}
 tr.active {{background: #fdd;}}
</style>
</head>
<body>
<h2>Episode replay</h2>
<p>Total steps: {len(steps)} | Final reward: {total_reward:.2f}</p>
<table id='replay'>
<thead><tr><th>Step</th><th>Action</th><th>Reward</th><th>Epsilon</th><th>Done</th><th>Cumulative</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<script>
const rows=document.querySelectorAll('#replay tbody tr');
let idx=0;const delay={int(speed*1000)};
function highlight(){{rows.forEach(r=>r.classList.remove('active'));if(idx<rows.length) rows[idx].classList.add('active');}}
highlight();
setInterval(()=>{{idx=(idx+1)%rows.length;highlight();}}, delay);
</script>
</body>
</html>"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML replay saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded Klondike episode")
    parser.add_argument("--file", required=True, help="CSV episode file")
    parser.add_argument("--speed", type=float, default=1.0, help="Delay between steps in seconds")
    parser.add_argument("--html", action="store_true", help="Also export an HTML replay")
    parser.add_argument("--export", action="store_true", help="Only export HTML without console output")
    args = parser.parse_args()

    if args.export:
        args.html = True

    steps = load_episode(args.file)

    if not args.export:
        replay_console(steps, args.speed)

    if args.html:
        base = os.path.splitext(os.path.basename(args.file))[0]
        html_path = os.path.join("replays", f"replay_{base}.html")
        episode_to_html(steps, html_path, args.speed)


if __name__ == "__main__":
    main()
