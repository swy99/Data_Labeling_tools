"""
시나리오/카메라별 레이블 현황 체크.

Usage:
    python check_status.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

DATA_ROOT = Path(__file__).parent / "data"


def main():
    counts = defaultdict(lambda: {"manual": 0, "auto": 0, "depth": 0, "total_imgs": 0})

    for cam_dir in sorted(DATA_ROOT.glob("*/Camera_*")):
        key = (cam_dir.parent.name, cam_dir.name)
        counts[key]["total_imgs"] = len(list(cam_dir.glob("*.jpg")))
        for jpath in cam_dir.glob("*.json"):
            try:
                d = json.load(open(jpath))
            except Exception:
                continue
            if not d.get("shapes"):
                continue
            if d.get("description") == "autolabel":
                counts[key]["auto"] += 1
            else:
                counts[key]["manual"] += 1
        depth_dir = cam_dir / "x-anylabeling-depth"
        if depth_dir.exists():
            counts[key]["depth"] = len(list(depth_dir.glob("*.npy")) + list(depth_dir.glob("*.npz")))

    print(f"{'시나리오':<20} {'카메라':<10} {'수동':>5} {'자동':>5} {'depth':>6} {'전체이미지':>10}")
    print("-" * 62)
    for key in sorted(counts):
        c = counts[key]
        flag = "  ← 필요" if c["manual"] == 0 and c["auto"] == 0 else ""
        print(f"{key[0]:<20} {key[1]:<10} {c['manual']:>5} {c['auto']:>5} {c['depth']:>6} {c['total_imgs']:>10}{flag}")

    total_manual = sum(c["manual"] for c in counts.values())
    total_auto = sum(c["auto"] for c in counts.values())
    total_depth = sum(c["depth"] for c in counts.values())
    total_imgs = sum(c["total_imgs"] for c in counts.values())
    print("-" * 62)
    print(f"{'합계':<30} {total_manual:>5} {total_auto:>5} {total_depth:>6} {total_imgs:>10}")


if __name__ == "__main__":
    import time
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            main()
            time.sleep(3)
    except KeyboardInterrupt:
        pass
