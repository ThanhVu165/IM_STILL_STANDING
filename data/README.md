# Organizer data directory

Bỏ toàn bộ dữ liệu do ban tổ chức cung cấp vào đây.

Suggested structure:

- `data/raw/videos/` : video gốc, `.mp4`, `.avi`, `.mov`, ...
- `data/raw/queries/` : file query / metadata / JSON / CSV nếu có
- `data/processed/keyframes/` : keyframe export, thumbnails, crop images
- `data/processed/embeddings/` : saved CLIP/SigLIP/ASR/OCR outputs
- `data/metadata/` : organizer metadata, annotations, ground truth
- `data/outputs/` : pipeline output manifests and generated result files

Lưu ý:
- Không commit dữ liệu lớn vào Git.
- Dùng các file `.gitkeep` để giữ cấu trúc thư mục.
- Khi chạy pipeline, bạn có thể trỏ video_path đến `data/raw/videos/...`.
