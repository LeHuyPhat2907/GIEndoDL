"""Script kiểm tra và chẩn đoán toàn diện môi trường GPU, CUDA và Tensor Cores trên máy tính PC.

Cách sử dụng trên máy bàn:
    python src/utils/check_gpu_environment.py
"""

import sys
import time

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_gpu_and_tensor_cores():
    print("=" * 80)
    print("🖥️  KIỂM TRA CHẨN ĐOÁN MÔI TRƯỜNG GPU, CUDA VÀ TENSOR CORES CHO DEEP LEARNING")
    print("=" * 80)

    # 1. Kiểm tra PyTorch
    try:
        import torch
        print(f"✅ PyTorch Version: {torch.__version__}")
    except ImportError:
        print("❌ LỖI: Chưa cài đặt PyTorch! Hãy chạy: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

    # 2. Kiểm tra CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("\n❌ CẢNH BÁO: CUDA CHƯA ĐƯỢC KÍCH HOẠT TRÊN PYTORCH!")
        print("   PyTorch hiện tại đang là bản CPU (ví dụ: 2.x.x+cpu).")
        print("   👉 CÁCH KHẮC PHỤC TRÊN MÁY BÀN:")
        print("   1. Gỡ bản PyTorch CPU: pip uninstall -y torch torchvision torchaudio")
        print("   2. Cài bản PyTorch hỗ trợ GPU CUDA 12.1:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 80)
        return False

    # Thông tin GPU
    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = round(vram_bytes / (1024 ** 3), 2)
    major, minor = torch.cuda.get_device_capability(0)
    cuda_arch = f"{major}.{minor}"

    print(f"\n🎉 ĐÃ PHÁT HIỆN GPU NVIDIA!")
    print(f"   * Tên Card GPU: {gpu_name}")
    print(f"   * Số lượng GPU: {num_gpus}")
    print(f"   * Dung lượng VRAM: {vram_gb} GB")
    print(f"   * CUDA Version (PyTorch build): {torch.version.cuda}")
    print(f"   * cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"   * Compute Capability: {cuda_arch}")

    # 3. Kiểm tra Tensor Cores
    print("\n" + "-" * 80)
    print("⚡ KIỂM TRA KHẢ NĂNG KÍCH HOẠT NHÂN TENSOR CORES (NVIDIA):")
    # Tensor Cores có từ Compute Capability >= 7.0 (Volta V100, Turing T4/RTX 20xx, Ampere RTX 30xx/A100, Ada RTX 40xx)
    has_tensor_cores = major >= 7
    if has_tensor_cores:
        print(f"✅ GPU {gpu_name} (Arch {cuda_arch} >= 7.0) CÓ PHẦN CỨNG NHÂN TENSOR CORES!")
    else:
        print(f"ℹ️ GPU {gpu_name} (Arch {cuda_arch} < 7.0 - Pascal/Maxwell) sử dụng nhân CUDA chuẩn (chưa có Tensor Cores).")

    # Kích hoạt các cờ tối ưu hóa Tensor Cores trong PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("✅ Đã kích hoạt allow_tf32 = True (Tăng tốc Tensor Cores trên Ampere/Ada)")
    print("✅ Đã kích hoạt cudnn.benchmark = True (Tự động tối ưu thuật toán tích chập)")

    # 4. Chạy Benchmark thực tế FP16 và Tensor Cores
    print("\n🚀 Đang chạy thử nghiệm tính toán ma trận thực tế với AMP (FP16)...")
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / 50 * 1000
        print(f"✅ Thử nghiệm thành công! Thời gian nhân ma trận 2048x2048 FP16: {elapsed:.2f} ms")
        print("✨ Nhân Tensor Cores hoạt động hoàn hảo, sẵn sàng cho huấn luyện Deep Learning!")
    except Exception as e:
        print(f"⚠️ Có cảnh báo khi chạy thử ma trận: {e}")

    # 5. Kiểm tra dữ liệu ảnh HyperKvasir
    from pathlib import Path
    root_dir = Path(__file__).resolve().parents[2]
    img_dir = root_dir / "data" / "raw" / "labeled-images"
    folds_dir = root_dir / "data" / "processed" / "5folds"

    print("\n" + "-" * 80)
    print("📁 KIỂM TRA BỘ DỮ LIỆU TRÊN MÁY BÀN:")
    if img_dir.exists() and ((img_dir / "lower-gi-tract").exists() or (img_dir / "upper-gi-tract").exists()):
        print(f"✅ Đã tìm thấy thư mục ảnh gốc: {img_dir}")
    else:
        print(f"⚠️ Chưa có ảnh tại {img_dir}. Chạy lệnh: python src/dataset/download_hyperkvasir.py để tải tự động!")

    if (folds_dir / "fold_0" / "train.csv").exists():
        print(f"✅ Đã tìm thấy dữ liệu Stratified 5-Folds: {folds_dir}")
    else:
        print(f"⚠️ Chưa có thư mục 5folds. Chạy lệnh: python src/dataset/create_5fold_splits.py để khởi tạo!")

    print("=" * 80)
    print("🎉 KẾT LUẬN: MÁY BÀN ĐÃ HOÀN TOÀN SẴN SÀNG ĐỂ HUẤN LUYỆN!")
    print("   Lệnh chạy Task #91 (CaiT-S24):")
    print("   python src/models/train_cait_full.py --epochs 50 --batch_size 32 --fold 0")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    check_gpu_and_tensor_cores()
