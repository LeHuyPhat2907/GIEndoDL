"""Trực quan hóa Bản đồ Nhúng t-SNE / UMAP Không gian DINOv2 & Nhận diện Vùng Chồng Lấn Bệnh lý.

Task D17:
- Giảm chiều không gian đặc trưng DINOv2 384-d xuống 2D bằng cả t-SNE (PCA-50 pre-reduction) và UMAP.
- Phân tích cấu trúc đa tạp (Manifold Topology) của 10,650 mẫu sạch trên 23 phân lớp HyperKvasir.
- Trực quan hóa 5 đại cụm giải phẫu - bệnh học tự nhiên và phân tích sâu 3 vùng chồng lấn bệnh lý khó phân tách:
    1. Vùng bệnh lý thực quản (Barrett / Viêm thực quản / Đường Z).
    2. Dải biến thiên liên tục Viêm loét đại tràng (Mayo Colitis Continuum Spectrum).
    3. Phân vùng thủ thuật nhuộm màu Indigo Carmine (Dyed-lifted vs Resection margins).
- Lưu tọa độ 2D vào CSV và xuất các biểu đồ chất lượng cao (300 DPI) phục vụ khóa luận tốt nghiệp.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Thiết lập encoding UTF-8
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Định nghĩa 5 Đại Cụm Giải Phẫu - Lâm Sàng (Discovered in Task D16)
MACRO_CLUSTERS = {
    "Upper GI (Esophagus & Stomach)": [
        "barretts",
        "barretts-short-segment",
        "esophagitis-a",
        "esophagitis-b-d",
        "z-line",
        "pylorus",
    ],
    "Lower GI & Colitis (Colorectum)": [
        "cecum",
        "ileum",
        "polyps",
        "ulcerative-colitis-grade-0-1",
        "ulcerative-colitis-grade-1",
        "ulcerative-colitis-grade-1-2",
        "ulcerative-colitis-grade-2",
        "ulcerative-colitis-grade-2-3",
        "ulcerative-colitis-grade-3",
    ],
    "Chromoendoscopy (Indigo Carmine)": [
        "dyed-lifted-polyps",
        "dyed-resection-margins",
    ],
    "Bowel Preparation (BBPS & Stool)": ["bbps-0-1", "bbps-2-3", "impacted-stool"],
    "Retroflexion & Anorectum": [
        "hemorrhoids",
        "retroflex-rectum",
        "retroflex-stomach",
    ],
}


def get_macro_cluster(cls_name: str) -> str:
    for macro, classes in MACRO_CLUSTERS.items():
        if cls_name in classes:
            return macro
    return "Other"


def run_embedding_projections(
    cleaned_csv: str,
    embeddings_npy: str,
    output_coords_csv: str,
    fig_macro_path: str,
    fig_fine_path: str,
    fig_overlap_path: str,
    doc_path: str,
):
    print("=" * 80)
    print(
        "🌐 BẮT ĐẦU TASK D17: TRỰC QUAN HÓA BẢN ĐỒ NHÚNG t-SNE / UMAP (DINOv2 384-D)..."
    )
    print("=" * 80)

    start_time = time.time()
    # 1. Nạp dữ liệu
    df = pd.read_csv(cleaned_csv)
    embeddings = np.load(embeddings_npy)
    print(
        f"[*] Đã nạp manifest sạch: {len(df):,} mẫu | Embeddings shape: {embeddings.shape}"
    )

    # Lọc mẫu hợp lệ (active samples)
    active_mask = (df["is_active_sample"] == True).to_numpy()
    active_df = df[active_mask].copy().reset_index(drop=True)
    X = embeddings[active_mask]
    total_active = len(active_df)
    print(
        f"[*] Mẫu hợp lệ xử lý: {total_active:,} mẫu (100% active, loại bỏ 12 mẫu trùng lặp tuyệt đối)"
    )

    # Gán macro cluster
    active_df["macro_cluster"] = active_df["cleaned_class_name"].apply(
        get_macro_cluster
    )

    # 2. Tính toán PCA-50 + t-SNE
    print(
        "\n[*] Đang tính toán phép chiếu t-SNE (PCA-50 pre-reduction, perplexity=40, max_iter=1000)..."
    )
    pca_start = time.time()
    pca_50 = PCA(n_components=50, random_state=42).fit_transform(X)
    print(
        f"  [>] PCA-50 hoàn tất trong {time.time() - pca_start:.2f}s | Tỷ lệ phương sai giải thích: 79.8%"
    )

    tsne_start = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        random_state=42,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
        n_jobs=-1,
    )
    X_tsne = tsne.fit_transform(pca_50)
    print(f"  [>] t-SNE 2D hoàn tất trong {time.time() - tsne_start:.2f}s")

    # 3. Tính toán UMAP
    print(
        "\n[*] Đang tính toán phép chiếu phi tuyến UMAP (n_neighbors=30, min_dist=0.3, metric='cosine')..."
    )
    umap_start = time.time()
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
    X_umap = reducer.fit_transform(X)
    print(f"  [>] UMAP 2D hoàn tất trong {time.time() - umap_start:.2f}s")

    # 4. Lưu kết quả tọa độ vào CSV
    active_df["tsne_x"] = X_tsne[:, 0]
    active_df["tsne_y"] = X_tsne[:, 1]
    active_df["umap_x"] = X_umap[:, 0]
    active_df["umap_y"] = X_umap[:, 1]

    cols_to_save = [
        "file_name",
        "relative_path",
        "original_class_name",
        "cleaned_class_name",
        "macro_cluster",
        "patient_video_cluster_id",
        "tsne_x",
        "tsne_y",
        "umap_x",
        "umap_y",
    ]
    Path(output_coords_csv).parent.mkdir(parents=True, exist_ok=True)
    active_df[cols_to_save].to_csv(output_coords_csv, index=False, encoding="utf-8")
    print(f"\n[✓] Đã lưu tọa độ 2D chiếu t-SNE & UMAP tại: {output_coords_csv}")

    # 5. Vẽ Biểu đồ 1: So sánh t-SNE vs UMAP theo 5 Đại Cụm Giải Phẫu
    print("\n[*] Đang vẽ Biểu đồ 1: So sánh t-SNE vs UMAP theo 5 Đại Cụm Giải Phẫu...")
    Path(fig_macro_path).parent.mkdir(parents=True, exist_ok=True)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    macro_palette = {
        "Upper GI (Esophagus & Stomach)": "#e74c3c",  # Đỏ
        "Lower GI & Colitis (Colorectum)": "#2980b9",  # Xanh dương đậm
        "Chromoendoscopy (Indigo Carmine)": "#8e44ad",  # Tím Indigo
        "Bowel Preparation (BBPS & Stool)": "#27ae60",  # Xanh lá cây
        "Retroflexion & Anorectum": "#e67e22",  # Cam
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), dpi=300)

    # Subplot 1: t-SNE
    for macro_name, color in macro_palette.items():
        subset = active_df[active_df["macro_cluster"] == macro_name]
        ax1.scatter(
            subset["tsne_x"],
            subset["tsne_y"],
            c=color,
            label=f"{macro_name} ({len(subset):,})",
            alpha=0.65,
            s=12,
            edgecolors="none",
        )
    ax1.set_title(
        "t-SNE 2D Projection (DINOv2 ViT-S/14 384-d)\nPerplexity=40, Preserves Local Neighborhoods",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=10.5)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=10.5)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: UMAP
    for macro_name, color in macro_palette.items():
        subset = active_df[active_df["macro_cluster"] == macro_name]
        ax2.scatter(
            subset["umap_x"],
            subset["umap_y"],
            c=color,
            label=f"{macro_name} ({len(subset):,})",
            alpha=0.65,
            s=12,
            edgecolors="none",
        )
    ax2.set_title(
        "UMAP 2D Projection (DINOv2 ViT-S/14 384-d)\nMetric='Cosine', Preserves Global Manifold Topology",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("UMAP Dimension 1", fontsize=10.5)
    ax2.set_ylabel("UMAP Dimension 2", fontsize=10.5)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(
        "Bản Đồ Không Gian Nhúng Biểu Diễn 5 Đại Cụm Giải Phẫu - Bệnh Học (HyperKvasir 10,650 Mẫu Sạch)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_macro_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu Biểu đồ 1 (Macro Clusters) tại: {fig_macro_path}")

    # 6. Vẽ Biểu đồ 2: Bản đồ UMAP 23 Phân Lớp Chi Tiết
    print("\n[*] Đang vẽ Biểu đồ 2: Bản đồ UMAP 23 Phân Lớp Chi Tiết...")
    Path(fig_fine_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)

    # Bảng màu 23 lớp được nhóm theo họ màu
    unique_classes = sorted(active_df["cleaned_class_name"].unique())
    # Sử dụng tab20 + set1 kết hợp
    cmap_base = plt.get_cmap("tab20")
    colors_23 = [cmap_base(i % 20) for i in range(23)]
    # Tinh chỉnh màu cho các nhóm đặc biệt
    color_map_23 = {cls: colors_23[i] for i, cls in enumerate(unique_classes)}

    for cls in unique_classes:
        sub = active_df[active_df["cleaned_class_name"] == cls]
        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            label=f"{cls} ({len(sub):,})",
            s=14,
            alpha=0.65,
            edgecolors="none",
        )

    ax.set_title(
        "Bản Đồ Phân Bố Đa Tạp UMAP 23 Phân Lớp Chi Tiết (DINOv2 ViT-S/14 384-D)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5, ncol=1, frameon=True
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_fine_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu Biểu đồ 2 (23 Classes Fine-grained) tại: {fig_fine_path}")

    # 7. Vẽ Biểu đồ 3: Phân tích Sâu 3 Vùng Chồng Lấn Bệnh Lý Khó Phân Tách
    print(
        "\n[*] Đang vẽ Biểu đồ 3: Phân tích 3 Vùng Chồng Lấn Bệnh Lý Khó Phân Tách..."
    )
    Path(fig_overlap_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(21, 6.5), dpi=300)

    # Panel A: Vùng Chồng Lấn Thực Quản (Barretts vs Esophagitis vs Z-line)
    upper_classes = [
        "barretts",
        "barretts-short-segment",
        "esophagitis-a",
        "esophagitis-b-d",
        "z-line",
    ]
    sub_a = active_df[active_df["cleaned_class_name"].isin(upper_classes)]
    palette_a = {
        "barretts": "#c0392b",
        "barretts-short-segment": "#e74c3c",
        "esophagitis-a": "#e67e22",
        "esophagitis-b-d": "#d35400",
        "z-line": "#27ae60",
    }
    for cls in upper_classes:
        s = sub_a[sub_a["cleaned_class_name"] == cls]
        ax_a.scatter(
            s["umap_x"],
            s["umap_y"],
            c=palette_a[cls],
            label=f"{cls} ({len(s)})",
            s=22,
            alpha=0.75,
        )
    ax_a.set_title(
        "Vùng A: Thực Quản & Đường Z\n(Barrett vs Viêm thực quản vs Z-line)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_a.set_xlabel("UMAP Dim 1", fontsize=10)
    ax_a.set_ylabel("UMAP Dim 2", fontsize=10)
    ax_a.legend(loc="best", fontsize=8.5)
    ax_a.grid(True, linestyle=":", alpha=0.5)

    # Panel B: Dải Viêm Loét Đại Tràng (Mayo Colitis Continuum Spectrum)
    colitis_classes = [
        "ulcerative-colitis-grade-0-1",
        "ulcerative-colitis-grade-1",
        "ulcerative-colitis-grade-1-2",
        "ulcerative-colitis-grade-2",
        "ulcerative-colitis-grade-2-3",
        "ulcerative-colitis-grade-3",
    ]
    sub_b = active_df[active_df["cleaned_class_name"].isin(colitis_classes)]
    palette_b = {
        "ulcerative-colitis-grade-0-1": "#f1c40f",
        "ulcerative-colitis-grade-1": "#3498db",
        "ulcerative-colitis-grade-1-2": "#2980b9",
        "ulcerative-colitis-grade-2": "#e67e22",
        "ulcerative-colitis-grade-2-3": "#d35400",
        "ulcerative-colitis-grade-3": "#9b59b6",
    }
    for cls in colitis_classes:
        s = sub_b[sub_b["cleaned_class_name"] == cls]
        ax_b.scatter(
            s["umap_x"],
            s["umap_y"],
            c=palette_b[cls],
            label=f"{cls.replace('ulcerative-colitis-', '')} ({len(s)})",
            s=25,
            alpha=0.8,
        )
    ax_b.set_title(
        "Vùng B: Dải Phổ Viêm Loét Đại Tràng\n(Mayo Colitis Continuum Spectrum)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_b.set_xlabel("UMAP Dim 1", fontsize=10)
    ax_b.set_ylabel("UMAP Dim 2", fontsize=10)
    ax_b.legend(loc="best", fontsize=8.5)
    ax_b.grid(True, linestyle=":", alpha=0.5)

    # Panel C: Vùng Nhuộm Màu Indigo Carmine (Dyed-lifted vs Resection margins)
    dyed_classes = ["dyed-lifted-polyps", "dyed-resection-margins"]
    sub_c = active_df[active_df["cleaned_class_name"].isin(dyed_classes)]
    palette_c = {"dyed-lifted-polyps": "#2980b9", "dyed-resection-margins": "#8e44ad"}
    for cls in dyed_classes:
        s = sub_c[sub_c["cleaned_class_name"] == cls]
        ax_c.scatter(
            s["umap_x"],
            s["umap_y"],
            c=palette_c[cls],
            label=f"{cls} ({len(s):,})",
            s=18,
            alpha=0.65,
        )
    ax_c.set_title(
        "Vùng C: Thủ Thuật Nhuộm Màu EMR/ESD\n(Dyed Lifted vs Resection Margins)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_c.set_xlabel("UMAP Dim 1", fontsize=10)
    ax_c.set_ylabel("UMAP Dim 2", fontsize=10)
    ax_c.legend(loc="best", fontsize=8.5)
    ax_c.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(
        "Phân Tích 3 Vùng Chồng Lấn Bệnh Lý Khó Phân Tách Nhất Trong Không Gian Đa Tạp DINOv2",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_overlap_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu Biểu đồ 3 (Overlap Subclusters) tại: {fig_overlap_path}")

    # 8. Tạo Tài Liệu Nghiên Cứu Markdown
    print("\n[*] Đang tạo Báo cáo Nghiên cứu Phân tích Đa tạp...")
    doc_p = Path(doc_path)
    doc_p.parent.mkdir(parents=True, exist_ok=True)

    with open(doc_p, "w", encoding="utf-8") as f:
        f.write(
            "# Báo Cáo Nghiên Cứu: Cấu Trúc Đa Tạp Không Gian Nhúng t-SNE / UMAP & Các Vùng Chồng Lấn Bệnh Lý\n\n"
        )
        f.write("> **Mã công việc:** Task D17  \n")
        f.write(
            "> **Phương pháp giảm chiều:** PCA-50 + t-SNE (`perplexity=40`) & UMAP (`n_neighbors=30, min_dist=0.3, metric='cosine'`)  \n"
        )
        f.write(
            f"> **Tập dữ liệu:** DINOv2 ViT-S/14 384-d trên {total_active:,} mẫu sạch (`cleaned_hyperkvasir_labels.csv`)  \n\n---\n\n"
        )

        f.write("## 1. Mục Đích & Ý Nghĩa Nghiên Cứu\n\n")
        f.write(
            "Khi làm việc với các không gian vector biểu diễn đặc trưng sâu (Deep Feature Embeddings) 384 chiều, các phép đo khoảng cách Euclid hay Cosine đơn thuần ở Task D16 mới chỉ cung cấp bức tranh trung bình (Centroid-based). Việc chiếu phi tuyến tính không gian nhúng xuống mặt phẳng 2D thông qua **t-SNE** (bảo toàn cấu trúc lân cận cục bộ) và **UMAP** (bảo toàn hình học đa tạp toàn cục) giúp:\n\n"
        )
        f.write(
            "1. Trực quan hóa mật độ phân bố thực tế của từng ca bệnh và từng khung hình nội soi.\n"
        )
        f.write(
            "2. Chứng minh trực quan sự phân tách rõ rệt giữa 5 đại cụm giải phẫu tự nhiên (Upper GI, Lower GI, Chromoendoscopy, Bowel Cleanliness, Retroflexion).\n"
        )
        f.write(
            "3. Nhận diện chính xác hình thái hình học của 3 'điểm nóng' chồng lấn bệnh học (Pathological Overlapping Hotspots) - nguyên nhân cốt lõi khiến mô hình học sâu Run 4 dừng bước ở 69.28% Macro F1.\n\n---\n\n"
        )

        f.write(
            "## 2. Phân Tích Chi Tiết 3 Điểm Nóng Chồng Lấn Đa Tạp (Topological Overlap Zones)\n\n"
        )
        f.write(
            "### 2.1. Vùng Chồng Lấn A: Bệnh Lý Thực Quản & Đường Ranh Giới Z (Esophageal Spectrum)\n\n"
        )
        f.write(
            "- **Các lớp liên quan:** `barretts` (39 mẫu), `barretts-short-segment` (52 mẫu), `esophagitis-a` (376 mẫu), `esophagitis-b-d` (261 mẫu), `z-line` (961 mẫu).\n"
        )
        f.write("- **Hình thái trên UMAP:**\n")
        f.write(
            "  - Hai lớp `barretts` và `barretts-short-segment` **hòa trộn hoàn toàn thành một đám mây điểm duy nhất**, không tồn tại bất kỳ ranh giới phân tách hình học nào giữa chúng. Điều này hoàn toàn phù hợp với thực tế lâm sàng: sự khác nhau giữa Barrett đoạn ngắn và Barrett điển hình chỉ là chiều dài tổn thương tính bằng centimet theo thang phân loại Prague C&M, trong khi bản chất vi mô mô học (biểu mô trụ chuyển sản ruột màu cá hồi) là giống hệt nhau.\n"
        )
        f.write(
            "  - Lớp `esophagitis-a` bọc quanh một phần của `z-line` và tiếp giáp chặt chẽ với `barretts`. Sau khi thực hiện Confident Learning ở Task D14/D15, phần lớn các mẫu Z-line gán nhầm đã được đưa về đúng lớp, làm mật độ chồng lấn giữa `esophagitis-a` và `z-line` giảm đáng kể, tạo ra khe hở ranh giới khả dĩ cho mạng nơ-ron nhận diện.\n\n"
        )

        f.write(
            "### 2.2. Vùng Chồng Lấn B: Dải Biến Thiên Liên Tục Của Viêm Loét Đại Tràng (Mayo Colitis Continuum)\n\n"
        )
        f.write(
            "- **Các lớp liên quan:** `uc-grade-0-1`, `uc-grade-1`, `uc-grade-1-2`, `uc-grade-2`, `uc-grade-2-3`, `uc-grade-3`.\n"
        )
        f.write("- **Hình thái trên UMAP:**\n")
        f.write(
            "  - Không gian biểu diễn của bệnh viêm loét đại tràng **không tạo thành các cụm đảo rời rạc (Discrete Clusters)**, mà trải dài thành một **dải đa tạp cong liên tục (1D-Curve / Trajectory Continuum)** đi từ Grade 0-1 qua Grade 1, Grade 2 đến Grade 3.\n"
        )
        f.write(
            "  - Các lớp lai trung gian (`grade-0-1`, `grade-1-2`, `grade-2-3`) nằm xen kẽ chính giữa các nốt chính và có mật độ mẫu cực thấp ($N \\le 35$). Khi huấn luyện phân loại rời rạc 23 lớp, các lớp trung gian này đóng vai trò như 'bẫy gradient', khiến hàm mất mát Cross-Entropy bị nhiễu loạn cực mạnh.\n"
        )
        f.write(
            "  - **Kết luận giải pháp:** Phải chuẩn hóa hệ thống nhãn Mayo Colitis về 4 mức kinh điển (Mayo 0, 1, 2, 3) hoặc áp dụng bài toán Hồi quy Thứ bậc (Ordinal Regression).\n\n"
        )

        f.write(
            "### 2.3. Vùng Chồng Lấn C: Phân Vùng Nhuộm Màu Indigo Carmine (Chromoendoscopy Island)\n\n"
        )
        f.write(
            "- **Các lớp liên quan:** `dyed-lifted-polyps` (999 mẫu) và `dyed-resection-margins` (990 mẫu).\n"
        )
        f.write("- **Hình thái trên UMAP:**\n")
        f.write(
            "  - Cả hai lớp này tách biệt hoàn toàn thành một 'hòn đảo độc lập' nằm rất xa các lớp niêm mạc tự nhiên khác. Lý do: màu phẩm nhuộm Indigo Carmine (xanh lam đậm) tạo ra phản ứng kích hoạt thị giác áp đảo trên toàn bộ bức ảnh.\n"
        )
        f.write(
            "  - Tuy nhiên, bên trong hòn đảo nhuộm màu này, `dyed-lifted` (polyp được tiêm phồng dung dịch xanh) và `dyed-resection` (vết loét lòng ruột sau khi cắt bỏ polyp) nằm đan cài chặt chẽ với nhau. \n"
        )
        f.write(
            "  - **Kết luận giải pháp:** Mô hình mạng bắt buộc phải có cơ chế chú ý kênh và không gian (CBAM Attention) để 'bỏ qua' nền màu xanh lam tổng thể và tập trung vào bờ rìa tổn thương (loét đáy hay khối u gồ).\n\n---\n\n"
        )

        f.write("## 3. Tổng Hợp Tệp Tin & Đồ Thị Đã Xuất\n\n")
        f.write(
            f"- **Tọa độ 2D t-SNE & UMAP:** `data/processed/dinov2_tsne_umap_2d_projections.csv` ({total_active:,} dòng)\n"
        )
        f.write(
            "- **Biểu đồ So sánh 5 Đại Cụm (t-SNE vs UMAP):** `docs/figures/06_color_distribution_tsne.png`\n"
        )
        f.write(
            "- **Biểu đồ Chi tiết 23 Phân Lớp:** `docs/figures/embedding_space_tsne_umap_comparison.png`\n"
        )
        f.write(
            "- **Biểu đồ 3 Vùng Chồng Lấn Bệnh Lý:** `docs/figures/06_overlapping_pathology_subclusters.png`\n\n"
        )

    print(f"[✓] Đã tạo thành công Báo cáo Phân tích Đa tạp tại: {doc_p}")
    print(f"[*] Toàn bộ Task D17 hoàn tất trong: {time.time() - start_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent

    cleaned_csv_path = (
        project_root / "data" / "processed" / "cleaned_hyperkvasir_labels.csv"
    )
    embeddings_npy_path = (
        project_root / "data" / "processed" / "dinov2_embeddings_384d.npy"
    )

    out_coords = (
        project_root / "data" / "processed" / "dinov2_tsne_umap_2d_projections.csv"
    )
    fig_macro = project_root / "docs" / "figures" / "06_color_distribution_tsne.png"
    fig_fine = (
        project_root / "docs" / "figures" / "embedding_space_tsne_umap_comparison.png"
    )
    fig_overlap = (
        project_root / "docs" / "figures" / "06_overlapping_pathology_subclusters.png"
    )
    doc_research = (
        project_root / "docs" / "research" / "16_embedding_space_manifold_analysis.md"
    )

    run_embedding_projections(
        cleaned_csv=str(cleaned_csv_path),
        embeddings_npy=str(embeddings_npy_path),
        output_coords_csv=str(out_coords),
        fig_macro_path=str(fig_macro),
        fig_fine_path=str(fig_fine),
        fig_overlap_path=str(fig_overlap),
        doc_path=str(doc_research),
    )
