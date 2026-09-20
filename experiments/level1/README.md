# Kết quả re-ranker đồ thị

## Kiểm tra bước 0

Mô hình mới khởi tạo được kiểm tra ở chế độ đánh giá trên đúng các ứng viên top-50 của tập val.
Các vector phần tử, điểm thô, z-score, điểm `Graph`, điểm trộn và thứ hạng của 200 truy vấn đầu
được đối chiếu với `experiments/checks/three_channel_probe_reference_scores.npz`, sai số tuyệt đối
tối đa cho phép là `1e-4`. R@1 toàn bộ val phải khớp `0.43496661337346` của phép thử ba kênh
không huấn luyện, với trọng số ban đầu `(a,b,c) = (0,60; 0,28; 0,12)` và nhiệt độ `0,05`.

Lệnh chạy:

```bash
~/venvs/mgrx/bin/python scripts/checks/step0_reference_check.py
```

Kết quả lần chạy kiểm tra được ghi trong log của script; kết quả huấn luyện sẽ bổ sung sau.
