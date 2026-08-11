# Script thuyết trình (tiếng Việt) — 10 phút, 11 slide chính

Quy ước: mỗi slide có (a) thời lượng gợi ý, (b) lời nói. Tổng ~9'40" để chừa margin
cho lúc lỡ nói chậm hoặc bị hỏi xen giữa. Số liệu trong script khớp đúng với số trên
slide — cứ đọc theo, không cần nhớ thêm.

---

## Slide title + Outline (~15s, nói khi đang chuyển slide, không cần dừng lại)

"Chào thầy/cô và các bạn. Nhóm em trình bày đề tài Robust and Sparse Mean-Variance
Portfolio Optimization, áp dụng trên rổ VN100. Em là Bình,
đây là Giang. Tụi em sẽ đi qua bài toán, cách giải, cách kiểm chứng lời giải, và kết
quả backtest ngoài mẫu."

---

## Slide 1 — The problem, in plain words (~55s)

"Bài toán rất đơn giản để phát biểu: nhà đầu tư có 1 đơn vị vốn, phải chia vào 98 trên
100 cổ phiếu trong rổ VN100 — 2 mã bị loại vì thiếu dữ liệu giao dịch đủ dài — dựa
trên 4 năm dữ liệu giá hàng ngày. Tỷ trọng mỗi cổ phiếu là bao nhiêu? Đây là bài toán
tối ưu danh mục kinh điển.

Markowitz mean-variance là điểm khởi đầu sách giáo khoa, nhưng có 3 câu hỏi tụi em
đặt ra để quyết định công thức tổng quát cần *có khả năng* bật/tắt cái gì — chứ
không phải bắt buộc phải có:

Một, ước lượng có đáng tin không? Sample mean trên cửa sổ ngắn là ước lượng rất tệ —
công thức có nên phòng hờ trường hợp xấu nhất không?

Hai, giữ cả 98 mã có đáng không? Mỗi vị thế tốn phí mở, theo dõi, đóng — công thức
có nên ưu tiên ít mã hơn không?

Ba, rủi ro thì luôn luôn quan trọng — đây là term duy nhất luôn bật trong mọi cấu
hình.

Và điều quan trọng: liệu từng lựa chọn đó có thực sự giúp ích hay không, tụi em
không giả định trước — sẽ kiểm chứng bằng thực nghiệm trên chính VN100, và câu trả
lời không hiển nhiên như trực giác."

---

## Slide 2 — Four objectives → four terms (~55s)

"Từ 3 câu hỏi đó, công thức tổng quát có 4 term, theo Chen, Ahipaşaoğlu, Zhang và
Yang 2024 — tụi em áp dụng framework này, không tự phát minh từ đầu.

Term thứ nhất, trừ mu-hat nhân w, tối đa hoá lợi nhuận kỳ vọng. Term thứ hai, kappa
nhân norm-2 của Sigma-mũ-một-phần-hai nhân w — đây là term robust, worst-case nếu
mu-hat sai, và quan trọng là term này được *suy ra* toán học chứ không phải thêm
tùy tiện. Term thứ ba, gamma nhân w chuyển vị Sigma w — chính là variance penalty cổ
điển của Markowitz. Term thứ tư, lambda nhân norm-1 của w — vừa là surrogate lồi cho
số lượng vị thế, vừa tự động chặn đòn bẩy gộp.

Một điểm cần lưu ý: không có ràng buộc w lớn hơn hoặc bằng 0 ở đây — bán khống được
phép trong mô hình tổng quát, đây là lựa chọn thiết kế có chủ đích, và tụi em sẽ
quay lại điểm này khi nói về biến thể long-only sau."

---

## Slide 3 — Type of problem ⇒ how we solve it (~55s)

"Bài toán này lồi: mỗi term hoặc là affine, hoặc là norm, hoặc là dạng toàn phương
PSD, và tập khả thi cũng affine — chứng minh chi tiết từng term ở phần backup. Lồi
nghĩa là mọi cực tiểu địa phương đều là cực tiểu toàn cục, nên một thuật toán bậc
nhất, cục bộ là đủ — tính lồi tự động nâng bất cứ điểm nào nó hội tụ tới thành
nghiệm tối ưu toàn cục.

Nhưng bài toán không trơn: kappa-norm không khả vi tại Sigma-mũ-nửa nhân w bằng 0,
và lambda-norm-1 không khả vi ở bất cứ đâu có w_i bằng 0 — tức đúng ngay tại nghiệm.
Gradient descent thường không áp dụng được.

Cách tụi em giải: tách phần không trơn ra, xử lý bằng bước prox. Phần g mượt thì lấy
subgradient, phần h không mượt — gồm cả lambda-norm-1 và ràng buộc ngân sách — thì
đưa vào phép chiếu prox. Ràng buộc tổng w bằng 1 nằm *bên trong* h, không xử lý
riêng."

---

## Slide 4 — The prox step: why it must be solved jointly (~50s)

"Bước prox chính là bài toán tối ưu con: cực tiểu nửa norm bình phương w trừ z, cộng
t-lambda norm-1 w, với ràng buộc tổng w bằng 1. Đưa nhân tử Lagrange nu vào ràng
buộc, bài toán tách được theo từng tọa độ: w_i của nu bằng soft-threshold của z_i
cộng nu. Chỉ cần tìm nu sao cho tổng các soft-threshold bằng 1 — hàm này liên tục và
không giảm, nên bisection với sai số 10 mũ trừ 12 là giải được. Tọa độ nào có trị
tuyệt đối nhỏ hơn ngưỡng thì về đúng 0.

Điểm tụi em muốn nhấn mạnh: đường tắt tưởng chừng hợp lý — soft-threshold trước, rồi
chiếu lên ràng buộc tổng bằng 1 sau — **không phải** là prox đúng của h. Phép chiếu
cộng thêm một hằng số vào *mọi* tọa độ, làm sống lại toàn bộ số 0 vừa tạo ra. Tụi em
đã thử cách này đầu tiên, và nó cho ra **không một số 0 nào cả**."

---

## Slide 5 — Does our solver actually work? (~55s)

"Tính lồi chỉ đảm bảo *bài toán* có một nghiệm tối ưu toàn cục — không đảm bảo
*code* của tụi em đúng. Một lỗi dấu hay bisection dừng sớm vẫn cho ra một w trông có
vẻ hợp lý, nhưng sai, mà không báo lỗi gì cả.

Nên tụi em kiểm chứng độc lập bằng CVXPY với solver CLARABEL — hai thuật toán không
dùng chung dòng code nào. Bảng bên trái là 6 cấu hình kappa, gamma, lambda khác
nhau: relative gap từ 10 mũ trừ 8 tới 10 mũ trừ 2, và Jaccard — tức tập vị thế active
trùng nhau — luôn là 1.00, nghĩa là active set giống hệt nhau ở mọi cấu hình. Chi phí
tại N bằng 98: prox mất 146 mili giây với 528 vòng lặp, CLARABEL chỉ 25 mili giây
với 14 vòng lặp — CLARABEL thắng khoảng 6 lần ở kích thước này, dù thứ tự sẽ đảo
ngược khi N lên tới hàng nghìn.

Dòng cuối bảng là trường hợp fail, và tụi em báo cáo thẳng thắn: khi kappa và lambda
đều bằng 0, cả hai term không trơn biến mất, bước step size alpha-0 trên căn k cộng
1 — vốn bắt buộc cho trường hợp không trơn — lại trở nên chậm không cần thiết trên
một hàm toàn phương giờ đã trơn. Nó dừng ở gap 2.0% sau 5000 vòng lặp."

---

## Slide 6 — From one formula to six named strategies (~40s)

"Từ một công thức tổng quát, tụi em định nghĩa 6 biến thể để test, mỗi biến thể chỉ
là công thức gốc với vài term cố định hoặc bỏ đi — không biến thể nào thêm gì mới
ngoài công thức đã có.

A là Robust Markowitz, long-only, tìm kappa. B là Classical Markowitz, long-only,
kappa bằng 0. C và D là hai benchmark thụ động: equal-weight và buy-and-hold, không
tối ưu gì cả. E là sparse-only, long-short, tìm lambda. F là công thức đầy đủ,
long-short, tìm cả kappa và lambda.

Với A và B, ràng buộc w lớn hơn 0 tự động làm tổng norm-1 của w bằng 1, nên lambda
bị loại khỏi lưới tìm kiếm của chúng thay vì tune vô ích — bất kỳ sự tập trung nào A,
B thể hiện là hiệu ứng *miễn phí* của hình học simplex, không phải do term sparsity
thiết kế."

---

## Slide 7 — Walk-forward out-of-sample backtest (~40s)

"Kết quả in-sample ở trên tối ưu trên dữ liệu đã thấy rồi — không nói lên gì về hiệu
năng khi triển khai thực tế. Nên tụi em backtest walk-forward.

Cửa sổ trượt 24 tháng: 18 tháng fit, 6 tháng validate, trượt 1 tháng mỗi lần — sơ đồ
cụ thể ở slide kế. Tái cân bằng hàng tháng, tổng 494 ngày giao dịch ngoài mẫu, 25 lần
tái cân bằng. Dùng Ledoit-Wolf shrinkage cho covariance, phí giao dịch 0.20% trên mỗi
đơn vị turnover. Hai benchmark C và D được xây từ cùng 98 mã trong rổ — không phải
chỉ số VN30 bên ngoài — để so sánh công bằng."

---

## Slide 8 — Rolling window, concretely (~35s)

"Đây là 3 chu kỳ liên tiếp cụ thể. Chu kỳ 1 dùng tháng 1 tới 18 để estimate, 19 tới
24 để validate, deploy ở tháng 25. Chu kỳ 2 trượt đúng 1 tháng — estimate 2 tới 19,
deploy tháng 26. Chu kỳ 3 tương tự, deploy tháng 27.

Điểm đáng chú ý: chu kỳ 1 và chu kỳ 2 dùng chung tới 23 trên 24 tháng dữ liệu — vậy
mà kappa, gamma vẫn đổi giữa hai chu kỳ, phần này tụi em bàn kỹ hơn ở backup. Cửa sổ
validation 6 tháng là một thước đo khá nhiễu, ngay cả khi dữ liệu bên dưới gần như
không đổi."

---

## Slide 9 — Results: all six variants (~65s, slide quan trọng nhất, nói chậm)

"Đây là kết quả chính. B — Classical Markowitz long-only — đạt cumulative return
74.2%, annualized 32.7%, Sharpe 1.05, cao nhất trong 6 biến thể. A robust thấp hơn:
37.7% cumulative, Sharpe 0.68. Hai benchmark C và D thấp hơn cả hai: quanh 15-21%
cumulative.

Và đây là phát hiện quan trọng nhất của bài: E và F — hai biến thể long-short không
giới hạn đòn bẩy — mất sạch 100% vốn. Nhưng nhìn cột Sharpe, E vẫn ghi 0.55, F ghi
0.42 — dương, trông không tệ. Đây là chỗ dễ hiểu lầm nhất: Sharpe ở đây là arithmetic
mean chia volatility, nhân căn 252 — **không phải** AnnRet chia Vol. Ở volatility
245%, volatility drag xấp xỉ một-nửa-sigma-bình-phương, tức khoảng 301%, vượt xa
mean cộng dồn khoảng 135%. Nói cách khác, Sharpe hoàn toàn **không nhìn thấy** đòn
bẩy — mà chính Sharpe lại là thước đo dùng để chọn tham số trong lưới tìm kiếm. Đây
là lý do E, F sụp đổ dù metric chọn tham số vẫn báo dương."

---

## Slide 10 — A concentration problem, and a constraint to fix it (~55s)

"A và B đã thắng cả hai benchmark, nhưng số mã nắm giữ dao động rất mạnh — B có
tháng chỉ giữ 1 mã, có tháng giữ 14 mã, trung bình 7.56. Tháng 2 năm 2025, toàn bộ
danh mục B dồn vào một mã duy nhất — VTP — rồi tháng sau lỗ 9.47% khi edge đó không
còn.

Giải pháp: thêm ràng buộc w_i nhỏ hơn hoặc bằng 1 trên K cho A và B, với K là số mã
tối thiểu nhà đầu tư chọn — ràng buộc này vẫn lồi. Với K bằng 5, tức trần 20% mỗi mã.

Kết quả: B sau khi capped đạt cumulative 76.3%, Sharpe tăng lên 1.39, đồng thời
volatility và drawdown giảm khoảng một phần ba. Cap này vừa **tăng** lợi nhuận vừa
**giảm** rủi ro — long-only sau khi thêm ràng buộc thắng rõ cả hai benchmark; còn
long-short không giới hạn đòn bẩy, như E, F, là bài học cảnh báo theo chiều ngược
lại — cơ chế cụ thể và ngày tệ nhất nằm ở phần backup."

---

## Slide 11 — Conclusion (~50s)

"Tổng kết lại. Một, tụi em xây dựng một mô hình lồi, biểu diễn được dưới dạng SOCP
theo Chen và cộng sự 2024, trong đó term robust được suy ra bằng toán chứ không giả
định, và term L1 làm hai việc cùng lúc: sparsity và chặn đòn bẩy.

Hai, giải bằng phương pháp proximal-subgradient, mà bước prox xử lý đồng thời và
chính xác cả penalty L1 lẫn ràng buộc ngân sách bằng bisection trên một nhân tử vô
hướng, nên các số 0 sau soft-threshold không bị hồi sinh. Tính lồi đảm bảo lời giải
cục bộ này là toàn cục, và CLARABEL xác nhận độc lập tới 6 chữ số có nghĩa.

Ba, ngoài mẫu: 6 cấu hình được backtest walk-forward. Long-only thắng cả hai
benchmark trên cơ sở risk-adjusted; ràng buộc tập trung theo mã đóng góp phần lớn lợi
thế đó. Term robust — kappa — **không chứng minh được giá trị**, cả trong lẫn ngoài
mẫu: chính thuật toán search của A tự chọn kappa bằng 0 ở 18 trên 25 chu kỳ.

Bốn, và đây là bài học rõ nhất: long-short không có trần đòn bẩy dẫn tới mất sạch
vốn — không chỉ rủi ro hơn, mà là một loại thất bại khác hẳn về bản chất.

Cảm ơn thầy/cô và các bạn đã lắng nghe, tụi em sẵn sàng trả lời câu hỏi."

---

## Ghi chú luyện tập

- Tổng thời gian lời nói ước tính: ~9 phút 40 giây (không tính khoảng dừng đổi
  slide, hỏi đáp giữa chừng). Còn dư khoảng 20 giây buffer trong khung 10 phút.
- Slide nặng nhất về nội dung là **Slide 9 (Results)** và **Slide 5 (Verify
  solver)** — nếu bị thiếu thời gian, có thể cắt bớt ở Slide 6 (Six methods, 40s
  → 25s, chỉ đọc tên 6 biến thể, bỏ đoạn giải thích lambda) và Slide 2 (bỏ câu
  "được suy ra chứ không tự phát minh").
- Nếu bị hỏi xoáy về Sharpe/volatility drag ở Slide 9 ngay lúc trình bày, có thể trả
  lời ngắn rồi hẹn "chi tiết em có ở slide backup" — slide Interpretation và
  Long-short wiped out đều có số liệu sẵn.
- Tất cả số liệu trong script này copy trực tiếp từ nội dung slide hiện tại
  (`slides/main.tex`) — nếu slide đổi số liệu sau này, phải update lại script.
