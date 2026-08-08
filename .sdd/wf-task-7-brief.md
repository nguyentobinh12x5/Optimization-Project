# Task Brief — Walk-Forward Backtest: Task 7 (final integration)

Đây là brief trích từ plan đầy đủ: `docs/superpowers/plans/2026-07-26-walk-forward-backtest.md`.

## Global Constraints (áp dụng cho toàn bộ plan, đọc kỹ)

- Spec đầy đủ: `docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md`.
- Solver long-short hiện tại (`solve()`, `portfolio_objective()`, `_prox_l1_simplex_eq`, `_robust_subgrad`) GIỮ NGUYÊN KHÔNG SỬA — đã CVXPY-verify, các phase trước phụ thuộc vào nó.
- Long-only: ràng buộc `w≥0, Σw=1`. Vì `‖w‖₁=Σw=1` là hằng số dưới ràng buộc này, **λ vô tác dụng** → KHÔNG đưa λ vào đường long-only.
- Rolling window: `lookback_months=24` (`E`=18 tháng đầu, `Vwin`=6 tháng cuối), rebalance **hàng tháng** (ngày giao dịch đầu tiên mỗi tháng).
- Param grid mặc định: κ ∈ {0, 0.5, 1, 2} × γ ∈ {1, 5, 10} = 12 tổ hợp. Chọn theo **Sharpe (rf=0)** trên `Vwin`.
- Shrinkage `"lw"` (Ledoit-Wolf) BẬT xuyên suốt backtest (cả lúc chọn tham số lẫn giải nghiệm cuối).
- Phí giao dịch mặc định `fee=0.002` (0.20%) × turnover mỗi kỳ; kỳ đầu turnover=1.
- KHÔNG look-ahead: mọi ước lượng/validation của kỳ `t` chỉ dùng dữ liệu có ngày < ngày rebalance `t`. Phải có test cấu trúc khẳng định điều này.
- Môi trường: `.venv/bin/python` (Python 3.14). KHÔNG scipy/sklearn — mọi thuật toán (kể cả simplex projection) viết bằng numpy thuần. Không pip install gì thêm (cvxpy đã có).
- Data thật: `data/returns.parquet` (965×98, đã sạch — xem `src/data_loader.py`).
- Project KHÔNG dùng git — KHÔNG chạy `git init`/`git add`/`git commit`. Ghi nhận hoàn tất từng task bằng cách append 1 dòng vào `.sdd/progress.md` (đúng quy ước ledger đã dùng xuyên suốt các phase trước của project).
- Style code/docstring: tiếng Việt/Anh nhất quán như các module hiện có, người đọc là dân tối ưu hoá.

---

### Task 7: Notebook integration + README

**Files:**
- Modify: `notebook.ipynb`
- Modify: `README.md`

**Interfaces:**
- Consumes: toàn bộ Task 1-6.

- [ ] **Step 1: Thêm section mới vào `notebook.ipynb`** (dùng `nbformat` để chèn cell, giống cách đã làm ở các phase trước — đọc file hiện có trước để chèn đúng vị trí, SAU section "Verify chéo CVXPY", TRƯỚC "Hạn chế & Kết luận"):

Markdown cell mở đầu section, giải thích:
- Vì sao cần OOS backtest (khác với phần in-sample trước đó — trả lời trực tiếp câu hỏi đã thảo luận trong hội thoại: toàn bộ phần trước ước lượng và tối ưu trên CÙNG dữ liệu).
- Cơ chế: rolling 24 tháng (18 estimation + 6 validation), rebalance hàng tháng, long-only (giải thích ngắn gọn vì sao λ bị loại — xem mục 2 của spec), tự chọn (κ,γ) qua Sharpe validation, phí 0.20%/turnover, benchmark equal-weight 1/N.

Code cell:
```python
from src.backtest import walk_forward_backtest, equal_weight_backtest, performance_metrics

strategy_result = walk_forward_backtest(returns)
benchmark_result = equal_weight_backtest(returns)

metrics_df = pd.DataFrame({
    "Chiến lược": performance_metrics(strategy_result.daily_returns),
    "Equal-weight 1/N": performance_metrics(benchmark_result.daily_returns),
}).T
display(metrics_df)
```

Code cell hiển thị 3 hình:
```python
from src.viz import fig7_backtest_equity, fig8_drawdown, fig9_selected_params

fig7_backtest_equity(strategy_result, benchmark_result)
plt.show()
fig8_drawdown(strategy_result, benchmark_result)
plt.show()
fig9_selected_params(strategy_result)
plt.show()
```

Markdown cell kết luận (viết TRUNG THỰC dựa trên số liệu THẬT sau khi chạy — KHÔNG viết trước kết luận rồi ép số liệu khớp): so sánh Sharpe/cumulative return/max_drawdown chiến lược vs 1/N, nêu rõ nếu KHÔNG vượt qua benchmark thì đó cũng là 1 kết quả hợp lệ và đáng báo cáo (1/N là baseline mạnh trong tài liệu portfolio optimization).

- [ ] **Step 2: Cập nhật mục "Hạn chế & Kết luận"** (cell cuối notebook): thêm rằng nay ĐÃ CÓ walk-forward OOS backtest (không còn thuần in-sample), nhưng còn hạn chế: chỉ ~24 tháng OOS (1 giai đoạn thị trường), model selection theo Sharpe trên validation 6 tháng cũng có thể overfit nhẹ, chưa multi-seed/bootstrap, chưa slippage.

- [ ] **Step 3: Chạy `nbconvert --execute` thật, xác nhận sạch**

```bash
.venv/bin/python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```
Expected: 0 lỗi cell. Nếu lỗi do thời gian chạy backtest quá lâu, xem lại Step 12 của Task 4 (giảm max_iter nếu cần) — SỬA Ở `src/backtest.py`, không sửa mù trong notebook.

- [ ] **Step 4: Verify không lỗi + đủ hình mới**

```python
import nbformat
nb = nbformat.read("notebook.ipynb", as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
imgs = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type")=="display_data" and "image/png" in o.get("data", {}))
errs = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type")=="error")
print(f"cells={len(nb.cells)} imgs={imgs} errors={errs}")
```
Expected: `errors=0`, `imgs` >= 9 (6 hình cũ + 3 hình mới).

- [ ] **Step 5: Cập nhật `README.md`** — thêm đoạn mô tả backtest walk-forward (module `src/backtest.py`, cách chạy, ý nghĩa) vào phần mô tả project; liệt kê 3 hình mới.

- [ ] **Step 6: Chạy toàn bộ test suite lần cuối**

```bash
.venv/bin/python -m pytest tests/ -v
```
Expected: PASS toàn bộ (test cũ + tất cả test mới từ Task 1-5).

- [ ] **Step 7: Ghi nhận hoàn tất task**

Project này KHÔNG dùng git (xem Global Constraints) — thay vì `git commit`, append 1 dòng vào `.sdd/progress.md` theo đúng quy ước ledger project đã dùng xuyên suốt các phase trước:
`Task <tên task>: complete (feat: integrate walk-forward backtest into notebook and README)`


---

## Số liệu THẬT đã verify độc lập (dùng để ĐỐI CHIẾU, không phải để chép vào notebook --
notebook phải tự TÍNH bằng code, số dưới đây chỉ để bạn phát hiện nếu kết quả bạn tính ra
lệch bất thường so với những gì đã biết là đúng)

Chạy `walk_forward_backtest(returns)` + `equal_weight_backtest(returns)` trên
`data/returns.parquet` (965x98) cho:

| Metric | Chien luoc | Equal-weight 1/N |
|---|---|---|
| cumulative_return | 0.2386 | 0.1472 |
| annualized_return | 0.1153 | 0.0726 |
| annualized_vol | 0.2868 | 0.1988 |
| sharpe | 0.5244 | 0.4523 |
| max_drawdown | -0.3789 | -0.2091 |
| n_days | 494 | 494 |

Tong phi (rebalance_log["cost"].sum()): chien luoc ~3.44%, 1/N ~0.49%.
Turnover trung binh chien luoc ~69%/thang; kappa duoc chon = 0 o hau het cac ky (validation
Sharpe 6-thang hiem khi uu ai robust term tren du lieu nay); gamma dao dong manh giua 1/5/10
giua cac ky lien tiep, gay turnover cao (6%-178%) va do bien dong lon hon 1/N.

KET LUAN TRUNG THUC can viet trong notebook (dua vao so THAT ban tu tinh ra, khong ep khac di):
chien luoc THANG ve cumulative return VA Sharpe, NHUNG danh doi bang rui ro cao hon dang ke
(vol +44%, max_drawdown gap ~1.8 lan) VA phi giao dich cao gap ~7 lan (turnover cao vi tham so
doi lien tuc qua validation nhieu). Day la mot DANH DOI risk-adjusted that, KHONG phai "thang"
hay "thua" don gian -- viet ca hai mat, dung tu "risk-adjusted trade-off" ro rang, KHONG chi noi
"chien luoc tot hon" ma bo qua phan rui ro/phi cao hon.
