import os
import zipfile
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
OUTPUT_ZIP = f"labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"


def main():
    label_files = list(DATA_DIR.rglob("*.json"))

    if not label_files:
        print("레이블 파일(.json)이 없습니다.")
        return

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for src in label_files:
            zf.write(src, src)  # data/scenario/Camera_N/xxx.json 경로 그대로 유지

    total = len(label_files)
    size_mb = os.path.getsize(OUTPUT_ZIP) / 1024 / 1024
    print(f"완료: {OUTPUT_ZIP} ({total}개 파일, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
