"""Script khởi chạy Web Application GIEndoDL CADx (1-Click Launch).

Tự động:
1. Khởi động máy chủ FastAPI (Uvicorn) tại cổng 8000.
2. Tự động mở trình duyệt web hiển thị giao diện lâm sàng.
3. Cung cấp đường dẫn truy cập mạng nội bộ (Localhost).
"""

import sys
import time
import webbrowser
import threading
from pathlib import Path

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def open_browser():
    """Tự động mở trình duyệt web sau khi server khởi động."""
    time.sleep(1.8)
    url = "http://127.0.0.1:8000"
    print(f"\n[🚀] Đang tự động mở trình duyệt tại: {url}\n")
    webbrowser.open(url)


def main():
    print("=" * 70)
    print("   🏥 GIEndoDL - HỆ THỐNG CADx HỖ TRỢ CHẨN ĐOÁN NỘI SOI TIÊU HÓA")
    print("   Đề tài Khóa Luận Tốt Nghiệp Cử Nhân / Kỹ Sư - ĐH Công Thương (HUIT)")
    print("   Mô hình: Hybrid CNN-CBAM-Transformer (Run 4 Production - 19 Classes)")
    print("=" * 70)
    print("\n[*] Đang khởi động máy chủ ứng dụng FastAPI...")

    # Mở trình duyệt song song trong luồng riêng
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        import uvicorn

        uvicorn.run(
            "web_app.backend.app:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n[!] Đã dừng máy chủ ứng dụng.")
    except Exception as e:
        print(f"\n[X] Lỗi khởi chạy: {e}")


if __name__ == "__main__":
    main()
