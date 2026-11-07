"""FastAPI Application Server cho Hệ Thống CADx Nội Soi Tiêu Hóa (GIEndoDL).

Cung cấp các API:
- POST /api/predict: Nhận ảnh tải lên hoặc ca mẫu -> Trả về kết quả chẩn đoán + Grad-CAM++
- GET /api/presets: Danh sách 6 ca bệnh mẫu tiêu biểu cho demo Hội đồng
- GET /api/health: Kiểm tra trạng thái mô hình và thiết bị tính toán
- Mount static files phục vụ giao diện Frontend
"""

from pathlib import Path
import sys
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from web_app.backend.model_engine import get_engine

app = FastAPI(
    title="GIEndoDL CADx Platform",
    description="Hệ thống Hỗ trợ Chẩn đoán Phân loại Tổn thương Bệnh lý Đường Tiêu hóa qua Nội soi",
    version="1.0.0",
)

# Kích hoạt CORS cho phát triển
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = PROJECT_ROOT / "web_app" / "frontend"
SAMPLE_DIR = FRONTEND_DIR / "sample_images"

# Mount thư mục ảnh mẫu
if SAMPLE_DIR.exists():
    app.mount(
        "/sample_images", StaticFiles(directory=str(SAMPLE_DIR)), name="sample_images"
    )


PRESET_CASES = [
    {
        "id": "case1_polyp",
        "title": "Ca 1: Polyp Đại trực tràng",
        "label": "polyps",
        "filename": "case1_polyp.jpg",
        "badge": "Tiền ung thư (Khẩn cấp)",
        "badge_color": "bg-red-500",
        "tract": "Lower GI (Đại trực tràng)",
        "description": "Khối u nhú lồi vào lòng ruột, cần chỉ định cắt polyp nội soi.",
    },
    {
        "id": "case2_ulcerative_colitis",
        "title": "Ca 2: Viêm loét đại tràng Mayo 2",
        "label": "ulcerative-colitis-grade-2",
        "filename": "case2_ulcerative_colitis.jpg",
        "badge": "Viêm tiến triển (Cảnh báo)",
        "badge_color": "bg-amber-500",
        "tract": "Lower GI (Đại tràng)",
        "description": "Niêm mạc sung huyết đỏ đậm, mất mạng mạch, mủn trợt dễ chảy máu.",
    },
    {
        "id": "case3_barretts",
        "title": "Ca 3: Barrett thực quản",
        "label": "barretts-esophagus",
        "filename": "case3_barretts.jpg",
        "badge": "Tiền ung thư (Khẩn cấp)",
        "badge_color": "bg-red-500",
        "tract": "Upper GI (Thực quản)",
        "description": "Chuyển sản tuyến niêm mạc thực quản do trào ngược mạn tính.",
    },
    {
        "id": "case4_esophagitis",
        "title": "Ca 4: Viêm thực quản trào ngược độ A",
        "label": "esophagitis-a",
        "filename": "case4_esophagitis.jpg",
        "badge": "Viêm nhẹ (Theo dõi)",
        "badge_color": "bg-amber-500",
        "tract": "Upper GI (Thực quản)",
        "description": "Vết trợt niêm mạc dưới 5 mm tại đường chuyển tiếp dạ dày thực quản.",
    },
    {
        "id": "case5_dyed_polyp",
        "title": "Ca 5: Polyp tiêm nâng & nhuộm màu",
        "label": "dyed-lifted-polyps",
        "filename": "case5_dyed_polyp.jpg",
        "badge": "Can thiệp thủ thuật",
        "badge_color": "bg-blue-500",
        "tract": "Lower GI (Đại tràng)",
        "description": "Polyp đã tiêm nâng dưới niêm mạc sẵn sàng cắt EMR/ESD.",
    },
    {
        "id": "case6_cecum_normal",
        "title": "Ca 6: Manh tràng bình thường",
        "label": "cecum",
        "filename": "case6_cecum_normal.jpg",
        "badge": "Mốc giải phẫu (Bình thường)",
        "badge_color": "bg-emerald-500",
        "tract": "Lower GI (Manh tràng)",
        "description": "Mốc giải phẫu đáy đại tràng bình thường, van hồi manh tràng rõ nét.",
    },
]


@app.on_event("startup")
async def startup_event():
    """Khởi động và tải trước mô hình AI."""
    print("[*] Khởi động FastAPI server...")
    _ = get_engine()
    print("[+] Mô hình đã sẵn sàng nhận yêu cầu chẩn đoán!")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Phục vụ trang chủ Frontend Single Page Application."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=404, detail="Không tìm thấy file frontend/index.html"
        )
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    """Kiểm tra tình trạng hoạt động của hệ thống."""
    engine = get_engine()
    return {
        "status": "healthy",
        "system": "GIEndoDL CADx Platform",
        "model_version": "Run 4 Optimal Production (19 Classes)",
        "device": str(engine.device),
        "num_classes": engine.num_classes,
    }


@app.get("/api/presets")
async def get_presets():
    """Trả về danh sách ca bệnh mẫu có sẵn."""
    return {"presets": PRESET_CASES}


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(None), preset_id: str = Form(None)):
    """API chính: Nhận ảnh từ tải lên hoặc mã ca mẫu -> Thực hiện chẩn đoán lâm sàng + Grad-CAM++."""
    engine = get_engine()

    try:
        pil_img = None

        if preset_id:
            # Tìm trong danh sách ca mẫu
            preset = next((p for p in PRESET_CASES if p["id"] == preset_id), None)
            if not preset:
                raise HTTPException(
                    status_code=400, detail=f"Không tìm thấy ca mẫu: {preset_id}"
                )
            img_path = SAMPLE_DIR / preset["filename"]
            if not img_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"File ảnh mẫu không tồn tại: {preset['filename']}",
                )
            pil_img = Image.open(img_path)

        elif file:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents))

        else:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp file ảnh hoặc mã ca mẫu (preset_id).",
            )

        # Thực thi mô hình và sinh giải thích XAI
        result = engine.predict(pil_img)
        return JSONResponse(content=result)

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích hình ảnh: {str(e)}")
