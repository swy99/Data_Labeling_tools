import os
import zipfile
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
OUTPUT_ZIP = f"labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"


def main():
    label_files = list(DATA_DIR.rglob("*.json"))
    depth_files = list(DATA_DIR.rglob("*.npy"))
    all_files = label_files + depth_files

    if not all_files:
        print("레이블/뎁스 파일이 없습니다.")
        return

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for src in all_files:
            zf.write(src, src)

    size_mb = os.path.getsize(OUTPUT_ZIP) / 1024 / 1024
    print(f"완료: {OUTPUT_ZIP}")
    print(f"  JSON: {len(label_files)}개 | NPY: {len(depth_files)}개 | 총 {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
