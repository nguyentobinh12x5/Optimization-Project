# SDD ledger — plan: PLAN.md (Sparse+Robust Portfolio Optimization)
# (lightweight, no-git mode — review by reading files)

Gate 1: complete (6 quyết định chốt 2026-07-25)

Task 1 (Phase 1 data layer): dispatched 2026-07-25 18:30 — implementer chạy nền (sonnet), fallback VCI→MSN→KBS, sleep 1.2s/req

Task 1 (Phase 1 data layer): complete (review clean — controller-verified)
  - src/data_loader.py, src/__init__.py, .gitignore(+data/)
  - data/prices.parquet (1090,75), data/returns.parquet (1069,75), vn100_symbols.csv (75 mã)
  - Verify: 100 mã tải, 25 loại → 75 sạch (≥70 ✓); window 2022-07-25→2026-07-24;
    returns 0 NaN; returns == pct_change(ffill(prices)) diff=0.0; cache run2 0.61s no-network ✓
  - Chất lượng code: không hardcode key, rate-limit 1.2s + fallback VCI→MSN→KBS + backoff,
    pct_change(fill_method=None) đúng chuẩn, hàm thuần testable. PASS đọc-review.
  - Note: tải thật mất 3116s (~52') do VCI timeout nhiều ở cuối — chi phí 1 lần, đã cache.
  - Note: subagent không ghi task-1-report.md (dừng khi đẩy download ra nền); controller đã verify thay.

Task 1 (Phase 1): REOPENED — controller đã over-claim "clean". Subagent report DONE_WITH_CONCERNS phát hiện:
  - MSN fallback chỉ trả ~365 phiên bất kể start → 23 mã (VIC,VHM,VNM,TCB,VRE,VIB,SHB,POW,PVD,PNJ,GVR...) bị
    gán nhãn listed_late SAI. Universe 75 mã thiếu nhiều blue-chip lớn nhất → bias nghiêm trọng cho μ̂/Σ.
  - Loại hợp lệ thật: DSE (niêm yết 2024-07), VPL (IPO 2025-05), GEE (thiếu >10%).
  - FIX cần: (a) patch loader — validate coverage, nếu rows < ~90% kỳ vọng thì retry VCI thay vì nhận MSN cắt cụt;
    bỏ MSN khỏi fallback cho lịch sử dài. (b) targeted re-fetch ~20 mã bị cắt cụt từ VCI, merge vào prices.
  - Phase 2 estimators đang chạy: CODE generic (không phụ thuộc mã) → vẫn hợp lệ; chỉ cần chạy lại số verify trên data đã sửa.

Task 2 (Phase 2 estimators): complete (review clean — controller-verified)
  - src/estimators.py, tests/test_estimators.py
  - estimate_mu/estimate_sigma(shrinkage None|'lw'|float)/matrix_sqrt_psd/estimate_all
  - pytest 8/8 pass (independent rerun); eigh+clip (không Cholesky), ép đối xứng, LW tự cài numpy
  - Code data-agnostic → không bị ảnh hưởng bởi fix universe Phase 1b; số verify trên data thật sẽ chạy lại sau 1b
  - Note: numpy RuntimeWarning benign (Apple Accelerate BLAS quirk), không NaN, rel_err Σ^½ ~1e-15

Task 1b (fix universe bias): dispatched — patch loader (bỏ MSN, retry VCI) + re-fetch ~25 mã, chạy nền

Task 1b (fix universe): complete (review clean — controller-verified)
  - Universe 75 → 97 mã; đủ VIC/VHM/VNM/TCB/VRE/VIB; loại đúng 3: DSE/VPL (late thật), GEE (>10% missing)
  - prices (1090,97), returns (1068,97), 0 NaN, diff vs pct_change=0.0, 2022-07-26→2026-07-24
  - Patch loader: QUOTE_SOURCE_FALLBACK=("VCI","KBS") bỏ MSN; retry VCI 3x backoff 5/15/30s; +repair_universe()
  - pytest estimators vẫn 8/8. Note máy: cần REQUESTS_CA_BUNDLE (Cloudflare Zero Trust proxy) cho network Python.

Task 1 (Phase 1): COMPLETE (thay cho lần over-claim trước) — data layer + universe 97 mã đại diện.
Task 2 (Phase 2): re-verified trên 97 mã — estimate_all OK, rel_err Σ^½=8.7e-16, đối xứng.

>>> NEXT: Gate 2 — trình bày thiết kế solver Phase 3 cho user review TRƯỚC khi code.

Task 3 (Phase 3 solver): DONE code (14/14 test) NHƯNG controller cross-check vs CVXPY lộ vấn đề thiết kế:
  - Solver hội tụ ĐÚNG hướng (gap giảm khi tăng iter) → subgradient/gradient đúng.
  - NHƯNG chậm: relgap objective ở 50k iter: (κ1γ5λ.01)=2.7%, (κ0γ5λ.1)=0.39%, (κ1γ5λ.001)=6.0% — mục tiêu Phase4 <0.5% FAIL 2/3.
  - QUAN TRỌNG: mất tính thưa. CVXPY active=9/16/38; hand active=48/97/45. Nguyên nhân: chiếu hyperplane
    w += (1-sum)/N·1 HỒI SINH các tọa độ vừa bị soft-threshold về 0 → không thể sparse.
  - FIX đề xuất (vẫn hand-written, không solver): thay "soft-threshold rồi chiếu đều" bằng JOINT PROX của
    (λ‖w‖₁ + I{1ᵀw=1}): w=soft_threshold(z+ν, αλ), tìm ν bằng bisection sao cho sum=1. Cho số 0 chính xác
    + feasible đồng thời, đúng nghĩa proximal-subgradient. → cần user quyết vì đổi thiết kế đã duyệt Gate 2.

Task 3 (Phase 3 solver): COMPLETE (fix-round joint-prox, review clean — controller cross-checked vs CVXPY)
  - _prox_l1_simplex_eq(z,t): soft-threshold + bisection ν → số 0 chính xác + sum=1 đồng thời
  - ALPHA0_DEFAULT retuned 0.1→10.0; max_iter 5000 đủ (hội tụ 232–2186 vòng)
  - pytest 16/16. Independent vs CVXPY (4 bộ tham số): conv=True, relgap ≤0.0007% (target <0.5% ✓),
    Jaccard active-set=1.000, exact-zeros 59–88, sum=1 tới 1e-8. Sparsity khớp CVXPY hoàn toàn.

Task 4 (Phase 4 CVXPY verify): COMPLETE (review clean — controller ran module)
  - src/cvxpy_check.py (cvxpy_solve/compare/DEFAULT_PARAM_GRID), tests/test_cvxpy_check.py
  - pytest 18/18. Bảng 6 bộ tham số: max|relgap|=0.000692% (<0.5% ✓), Jaccard=1.0 trên 5 bộ thưa,
    λ=0 bộ = 0.99 (đúng kỳ vọng, không thưa), tất cả converged, không cần fallback SCS.
  - cvxpy KHÔNG leak vào solver (match grep chỉ là docstring "KHÔNG dùng cvxpy"). Constraint giữ.

>>> Phase 1-4 XONG. NEXT: Phase 5 viz (Gate 3 nhẹ — user chọn hình). Rồi Phase 6 notebook+README.

Task 5 (Phase 5 viz): COMPLETE code (review clean — controller đã xem 5/6 hình)
  - src/viz.py (6 hàm + generate_all), figures/fig1..6.png; theo skill dataviz (palette validated)
  - Hình chất lượng tốt: nhãn+đơn vị+legend/colorbar, style nhất quán, regenerate sạch từ cache 9s
  - Finding thật (fig5): robust term κ>0 LÀM TĂNG cả active-set lẫn HHI (phản trực giác) — giữ nguyên, bàn ở notebook
  - >>> Gate 3: đang hỏi user chọn hình nào vào notebook

Task 6 (Phase 6 notebook+README): COMPLETE (review clean — controller executed nb)
  - notebook.ipynb (22 cell, 11 code, 6 hình inline, 0 lỗi, nbconvert --execute sạch ~15s, no network)
  - README.md (đầy đủ), + controller thêm requirements.txt (pinned versions)
  - pytest 18/18. Docstring solver mô tả đúng joint-prox (prox-then-project chỉ là history).

>>> Final whole-branch review: dispatched (opus), đọc toàn bộ src/tests/notebook/README.

Final review (opus): KHÔNG Critical, KHÔNG lỗi correctness. Solver verify chéo CVXPY OK, không rò rỉ.
  - I1 (README claim requirements.txt + thiếu pyarrow/vnai): FIXED — README trỏ requirements.txt, thêm pyarrow, bỏ seaborn.
  - M1 (docstring fallback VCI→MSN→KBS lệch): FIXED (data_loader → VCI→KBS, ghi MSN loại).
  - M2 (import thừa pytest/SolveResult): FIXED (test vẫn 18/18).
  - M3 (viz docstring đơn vị relgap): FIXED.
  - M6 (seaborn thừa): FIXED (bỏ khỏi requirements.txt, không import đâu).
  - M4/M5/M7 (docstring best_obj edge w0 / LW ddof non-default / tên 'converged'): DEFERRED — Minor,
    non-load-bearing, defensible; ghi lại làm known-notes, không chặn deliverable.
  - Re-verify sau fix: pytest 18/18, all modules import OK. Logic KHÔNG đổi (chỉ doc/README/import).

=== PROJECT COMPLETE: Phase 1-6 xong, final review sạch. Deliverable: notebook.ipynb + README + requirements.txt
    + src/ (5 module) + tests/ (18 test) + figures/ (6 hình) + data cache. ===

Task 7 (FIX: weekend duplicate rows): dispatched, chạy nền.
  - Root cause: ~71 "phiên Chủ nhật ma" trong prices.parquet (giá trùng phiên liền trước, gắn sai ngày cuối tuần).
    68/1068 dòng returns.parquet (6.4%) có return=0.0 CHÍNH XÁC mọi mã cùng lúc -- méo mu_hat (kéo về 0) +
    Sigma (thổi phồng tương quan chéo giả).
  - Fix code: clean_prices() thêm `prices.loc[prices.index.dayofweek < 5]` NGAY sau cắt [start,end], TRƯỚC
    khi tính missing_frac/first_valid (để mẫu số đúng, quyết định loại/giữ mã tính lại chuẩn).
  - Backup pre-fix: /tmp/prices_before_fix.parquet, /tmp/returns_before_fix.parquet (để so sánh định lượng).
  - Subagent: FORCE_REFRESH=1 re-fetch toàn bộ (không tái dùng cache cũ nhiễm lỗi) + verify + so sánh
    mu_hat/Sigma trước-sau + rerun pytest/cvxpy_check/viz/notebook + update văn bản nếu số đổi.

Task 7 SỰ CỐ: lần fetch đầu (subagent, force_refresh) THẤT BẠI NẶNG -- cache bị ghi đè bằng
  (1994, 0): 0 mã giữ lại, mọi mã "50% missing". Đã khôi phục ngay từ /tmp/*_before_fix.parquet
  để không để repo ở trạng thái hỏng.
  ROOT CAUSE (đã xác nhận bằng test trực tiếp): VCI trả timestamp 00:00:00, KBS trả 07:00:00 cho
  CÙNG 1 ngày giao dịch -- pd.concat(axis=1) coi là 2 index khác nhau. Chỉ cần 1 mã fallback KBS
  (như TCH lần này) là đủ làm UNION INDEX GẦN NHƯ NHÂN ĐÔI (997->1994) và mọi mã VCI khác bị NaN giả
  ~50% ở các dòng "giờ lạ". Bug này LATENT từ đầu vì KBS gần như chưa từng thực sự được dùng ở các lần
  chạy trước (VCI retry luôn thành công).
  FIX: fetch_history_one_symbol() thêm `.dt.normalize()` sau parse timestamp (chuẩn hóa mọi source về
  cùng giờ 00:00:00) + dedup index trong-1-mã (keep last). Verified bằng test trực tiếp: VCI+KBS merge
  giờ cho đúng 997 hàng, 0 NaN.
  Đang chạy lại full fetch LẦN 3 (refetch2.log, PID theo dõi qua bq7lf9clw), cả 2 fix (weekend + VCI/KBS
  timestamp) cùng có hiệu lực.

Task 7 HOÀN TẤT (lần 3, thành công) -- controller tự verify toàn bộ, không qua subagent:
  - Fetch mới: 100/100 mã tải được, 2 loại (DSE/VPL niêm yết muộn thật), universe 98 mã.
  - prices (997,98), returns (965,98). Verify: 0 phiên cuối tuần, 0 duplicate index, 0 NaN,
    đủ blue-chip, vn100_symbols.csv khớp returns.columns, consistency diff=0.0.
  - Định lượng tác động fix (so /tmp backup cũ, 97 mã chung): variance trung bình TĂNG ~8.9%
    (4.997e-4 -> 5.440e-4, đúng hướng dự đoán vì loại bỏ "ngày phẳng giả"); tương quan trung
    bình off-diag giảm nhẹ 0.3575->0.3551; mu_hat |trung bình| tăng nhẹ 0.000476->0.000561.
  - pytest 18/18, cvxpy_check: max relgap 0.000775% (<0.5%), Jaccard=1.0 TRÊN CẢ 6/6 bộ
    (trước đây bộ lambda=0 chỉ 0.99, giờ khớp tuyệt đối nhờ data sạch hơn).
  - Figures regenerate sạch (6/6).
  - BUG THỨ 3 phát hiện + fix ngay: notebook.ipynb kernelspec.name="python3" trỏ nhầm sang
    kernel hệ thống KHÔNG LIÊN QUAN (Python 3.13 global, thiếu cvxpy) thay vì kernel đúng
    "optproj-venv" (trỏ .venv project) mà Phase 6 đã đăng ký nhưng quên gán vào notebook.
    Fix: sửa kernelspec.name -> "optproj-venv". Re-execute sạch 23 cell, 0 lỗi.
  - Đã cập nhật số liệu cứng lỗi thời trong notebook (97->98 mã ở 3 chỗ) + README (97->98) +
    thêm đoạn markdown giải thích 2 lỗi dữ liệu đã fix (cuối tuần + timestamp VCI/KBS).
  - Đã dọn backup tạm /tmp/*_before_fix.parquet sau khi xác nhận ổn định.

Sự cố quy trình đáng ghi nhớ: subagent lần 1 của Task 7 tải xong nhưng gây (1994,0) do bug
  timestamp VCI/KBS chưa biết lúc đó -- controller đã khôi phục backup kịp thời, không mất dữ
  liệu. Lần chạy lại (subagent lần 2) rơi vào vòng lặp tự-poll vô ích (tool_uses tăng dần mỗi
  lần notify mà không tiến triển) -- đã TaskStop nó, chuyển sang controller tự quản lý trực
  tiếp (bash nền + wait-loop riêng) cho phần còn lại của Task 7, hiệu quả hơn.

=== PROJECT lại COMPLETE với dữ liệu ĐÃ SỬA (2 lỗi mới + 1 lỗi kernelspec), verify độc lập toàn bộ. ===

Task 8 (đề xuất user): thêm bộ lọc thống kê "ngày lễ giả" cho ngày lễ VN rơi vào NGÀY THƯỜNG
  (Tết, 30/4, 1/5, 2/9... -- dayofweek>=5 không bắt được). Điều tra trước: 0 ngày hiện tại có
  >=90% mã return=0.0 (max quan sát 23.4%); đối chiếu 15 ngày lễ cố định 2023-2026 -- TẤT CẢ đã
  vắng mặt sẵn trong prices.index (VCI xử lý đúng lịch lễ, không bị bug như cuối tuần).
  => Fix KHÔNG đổi dữ liệu hiện tại (verified: compute_returns mới cho identical output với cache
  cũ), nhưng thêm như lớp phòng vệ cho các lần tải lại sau.
  Code: compute_returns() thêm tham số holiday_zero_frac=0.90 (HOLIDAY_ZERO_RETURN_FRAC hằng số);
  loại phiên có >=90% mã (trong số mã có data) return=0.0 chính xác, log rõ ngày bị loại.
  Test mới: tests/test_data_loader.py (4 test: weekend filter, holiday-artifact detection với
  synthetic data, KHÔNG loại nhầm ngày đứng giá thật thiểu số 17%, không NaN sau lọc).
  pytest: 22/22 pass (18 cũ + 4 mới). Notebook cập nhật đoạn giải thích lớp phòng vệ này (minh
  bạch: hiện chưa loại thêm phiên nào), re-execute sạch 23 cell/0 lỗi.

======================================================================
NEW PLAN: docs/superpowers/plans/2026-07-26-walk-forward-backtest.md
Subagent-driven development, no-git mode (như các phase trước).
Naming: brief/report dùng prefix wf-task-N (phân biệt với task-N cũ của 8 phase gốc).
======================================================================

wf-Task 1: complete (feat: add simplex_projection Duchi 2008 + portfolio_objective_long_only + solve_long_only vao src/prox_solver.py; tai su dung _smooth_subgrad/_robust_subgrad va SolveResult co san khong sua; test moi tests/test_prox_solver.py 7 test (4 simplex_projection + 3 solve_long_only); pytest tests/test_prox_solver.py: 15/15 pass (8 cu + 7 moi))

wf-Task 1: COMPLETE (review clean — controller verified: 29/29 pytest, solve() cũ nguyên vẹn,
  solve_long_only tái dùng _smooth_subgrad có sẵn thay vì chép công thức, convergence logic khớp solve()).
  Note: phát hiện .git/ tồn tại thật (staged, chưa commit, tạo lúc 14:38) -- không đụng vào, chỉ ghi nhận.

wf-Task 2: complete (feat: verify long-only solver against CVXPY; them cvxpy_solve_long_only + compare_long_only vao src/cvxpy_check.py (khong co term lam*||w||_1, dung w>=0,sum(w)=1); 2 test moi trong tests/test_cvxpy_check.py (real data 965x98 shrinkage=lw + small synthetic); relgap that: max ~0.086% (kappa=0,gamma=1, chua converge trong max_iter mac dinh) toi 3.3e-6% (kappa=2,gamma=10), tat ca <0.5%, jaccard=1.0 moi bo tham so; pytest -q: 31/31 pass)

wf-Task 2: COMPLETE (review clean — controller verified: 31/31 pytest, cvxpy_solve_long_only
  formulation đúng w>=0/sum=1 KHÔNG lam, không leak cvxpy vào solver). relgap max 0.086%, Jaccard=1.0.
wf-Task 3: complete (feat: add performance_metrics for backtest evaluation vao src/backtest.py moi; tests/test_backtest.py 3 test theo brief; fix nho so voi code mau brief: pandas Series.std tren chuoi hang so co the ra ~1e-19 (khong dung 0.0 tuyet doi) do sai so lam tron -> them STD_ZERO_EPS=1e-12 nguong thay vi so sanh > 0 thuan, tranh sharpe bung no thay vi NaN; pytest full suite: 34/34 pass (31 cu + 3 moi))

wf-Task 3: COMPLETE (review clean — controller verified 34/34 pytest). Bắt được edge case thật:
  std(chuỗi hằng số) không đúng 0.0 tuyệt đối (sai số float ~2e-19) -> Sharpe bùng nổ nếu dùng
  `std>0` thô. Fix: STD_ZERO_EPS=1e-12 threshold. Đây là fix hợp lệ so với brief gốc.

wf-Task 4: complete (feat: implement walk-forward backtest engine — them _build_rebalance_windows (no-look-ahead structural test), _simulate_period (drift + turnover + fee), BacktestResult + walk_forward_backtest vao src/backtest.py; consume solve_long_only/estimate_all/performance_metrics; TDD tung ham (FAIL->implement->PASS); pytest tests/: 39/39 pass (34 cu + 5 moi); smoke test data that (965x98): 28.81s, 25 ky rebalance, 0 NaN, turnover ky dau=1.0, weights feasible (sum=1, w>=0))

wf-Task 4: COMPLETE (review clean — controller verified: 39/39 pytest, no-look-ahead logic đọc
  code xác nhận đúng, smoke test độc lập trên data thật: 25 kỳ, 28.3s, 0 NaN, weights feasible).
  Finding thật (không phải bug): kappa=0 được chọn phần lớn (Sharpe validation 6-tháng ít khi ưu
  ái robust term); n_active dao động 2-17/98, turnover một số kỳ >1.0 (tối đa lý thuyết 2.0 cho
  long-only) do (kappa,gamma) đổi mạnh giữa các kỳ -- đặc điểm thật của mẫu validation nhỏ, cần
  diễn giải trung thực ở Task 7 notebook.

wf-Task 5: complete (feat: add equal_weight_backtest benchmark vao src/backtest.py, tai su dung _build_rebalance_windows/_simulate_period/BacktestResult co san, trong so 1/N moi ky, cung danh sach ngay rebalance voi walk_forward_backtest de so sanh cong bang tren cung OOS period; test moi trong tests/test_backtest.py theo brief; pytest tests/: 40/40 pass (39 cu + 1 moi))

wf-Task 5: COMPLETE (review clean — controller verified 40/40 pytest + smoke test thật:
  25 kỳ khớp walk_forward, weights uniform, 0 NaN. Benchmark 1/N thật: cumret=14.7%,
  Sharpe=0.45, max_drawdown=-20.9% qua ~2 năm OOS -- số tham chiếu cho Task 7 so sánh.

wf-Task 6: complete (feat: add backtest visualization (equity curve, drawdown, selected params); them fig7_backtest_equity/fig8_drawdown/fig9_selected_params vao src/viz.py, tai su dung _apply_style()/_save()/CAT_COLORS co san tu fig1-6, khong dua vao generate_all vi walk_forward_backtest ton thoi gian; smoke test data that (965x98): 3 PNG sinh thanh cong trong figures/, 0 NaN tren 494 ngay OOS, kappa/gamma/turnover hien thi ro; pytest tests/: 40/40 pass khong regressions)

wf-Task 6: COMPLETE (review clean — controller verified 40/40 pytest + xem trực tiếp 3 hình + tính
  lại bảng metrics đầy đủ). KẾT QUẢ THẬT (không phải lỗi, cần diễn giải trung thực ở Task 7):
  Chiến lược: cumret=23.9% annret=11.5% vol=28.7% Sharpe=0.524 maxDD=-37.9% phí=3.44%
  1/N:        cumret=14.7% annret=7.3%  vol=19.9% Sharpe=0.452 maxDD=-20.9% phí=0.49%
  => Chiến lược thắng return+Sharpe nhưng rủi ro/phí cao hơn nhiều (turnover TB 69%/tháng do
  (kappa,gamma) đổi liên tục qua validation Sharpe 6-tháng nhiễu, kappa≈0 hầu hết các kỳ).
  Đây là đánh đổi risk-adjusted thật, KHÔNG phải "thắng" hay "thua" đơn giản.

wf-Task 7: COMPLETE (subagent rớt kết nối giữa chừng do lỗi hạ tầng, không phải lỗi task --
  đã hoàn thành phần việc nặng nhất: notebook section + kết luận trung thực + nbconvert sạch
  lần đầu. Controller tự hoàn tất phần còn thiếu: đồng bộ README mục pytest (18->40 test) và
  mục mô tả notebook (22 cell/6 hình/~15s -> 28 cell/9 hình/~50s), viết report, ghi ledger).
  Verify cuối: pytest 40/40, nbconvert 0 lỗi (28 cell, 9 hình, ~50s).

=== WALK-FORWARD BACKTEST PLAN HOÀN TẤT (7/7 task) ===

Final review walk-forward backtest (opus): SẠCH — 0 Critical, 0 Important, 5 Minor (đa số
  "by design" hoặc ghi chú, không phải bug). Đã dọn 1 Minor (logger unused trong backtest.py)
  vì rủi ro=0. pytest 40/40 sau dọn dẹp.
  4 Minor còn lại (không sửa, defensible): val_sharpe=-inf fallback khi mọi param NaN (không
  xảy ra trên data thật); Sharpe validation dùng w cố định không drift/phí (ĐÚNG theo spec §3);
  generate_all() không gồm fig7-9 (by design, cần BacktestResult); vòng validation là điểm
  chậm nhất (không ảnh hưởng correctness).

=== WALK-FORWARD BACKTEST FEATURE HOÀN TẤT VÀ VERIFY SẠCH (7/7 task + final review). ===
