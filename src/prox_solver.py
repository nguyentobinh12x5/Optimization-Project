"""
src/prox_solver.py
====================

Solver proximal-subgradient TỰ VIẾT (numpy thuần) cho bài toán sparse+robust
portfolio optimization:

    min_w  f(w) = -mu^T w + kappa * ||Sigma^(1/2) w||_2 + gamma * w^T Sigma w
                  + lam * ||w||_1
    s.t.   1^T w = 1

w KHÔNG bị ràng buộc không âm (cho phép bán khống). KHÔNG dùng cvxpy,
scipy.optimize, sklearn hay bất kỳ solver có sẵn nào -- toàn bộ thuật toán
cài đặt bằng numpy thuần theo thiết kế đã duyệt (Gate 2).

Người đọc mục tiêu là dân TỐI ƯU HOÁ. Phase 4 verify chéo kết quả bằng
CVXPY; bản FIX (xem mục "FIX-ROUND: joint prox" bên dưới) đã sửa lỗi
prox-then-project của bản đầu (bị cross-check CVXPY phát hiện phá sparsity)
bằng joint prox chính xác qua bisection -- số liệu so sánh với CVXPY sau
fix nằm trong task-3-report.md.

---------------------------------------------------------------------------
Công thức subgradient (điểm mấu chốt, dễ sai)
---------------------------------------------------------------------------
f(w) có 3 thành phần không trơn / khó vi phân trực tiếp: term robust
`kappa*||Sigma^(1/2) w||_2` (không khả vi tại w=0, nhưng thường w != 0 nên
khả vi hầu khắp nơi) và term `lam*||w||_1` (không khả vi tại w_i=0, xử lý
riêng bằng prox, xem bên dưới). Phần còn lại (`-mu^T w + gamma*w^T Sigma w`)
khả vi trơn thông thường.

Đạo hàm của `||Sigma^(1/2) w||_2` theo w: đặt u = Sigma^(1/2) w. Với g(w) =
||u||_2 = sqrt(u^T u) = sqrt(w^T Sigma^(1/2) Sigma^(1/2) w) = sqrt(w^T Sigma
w) (vì Sigma^(1/2) đối xứng nên (Sigma^(1/2))^T Sigma^(1/2) = Sigma^(1/2)
Sigma^(1/2) = Sigma). Do đó:

    d/dw ||u||_2 = (Sigma w) / ||u||_2 = (Sigma w) / ||Sigma^(1/2) w||_2

QUAN TRỌNG: tử số là `Sigma @ w` (dùng Sigma đầy đủ), KHÔNG PHẢI
`Sigma^(1/2) @ w` -- nhầm giữa hai cái này là lỗi phổ biến. Lý do: chain
rule qua u = Sigma^(1/2) w cho ra Jacobian (Sigma^(1/2))^T = Sigma^(1/2) (vì
đối xứng), rồi (Sigma^(1/2))^T @ u = Sigma^(1/2) @ (Sigma^(1/2) @ w) =
Sigma @ w.

Khi ||Sigma^(1/2) w||_2 <= eps (w gần 0 theo chuẩn Sigma^(1/2), về lý thuyết
chỉ đúng đại số nếu w=0 do Sigma^(1/2) PSD), hàm không khả vi -- ta chọn
subgradient = 0 (một phần tử hợp lệ của tập dưới vi phân tại điểm kỳ dị,
vì 0 luôn thuộc dưới vi phân của norm tại gốc). Điều này tránh chia cho 0 /
NaN. Xem `_robust_subgrad`.

---------------------------------------------------------------------------
Thuật toán mỗi vòng lặp k (đã duyệt, bản FIX sau cross-check CVXPY --
xem "FIX-ROUND: joint prox" bên dưới để biết lý do đổi bước 3-4)
---------------------------------------------------------------------------
1. Subgradient phần trơn + robust (KHÔNG gồm L1 -- L1 xử lý bằng prox):
       v = -mu + 2*gamma*Sigma@w_k + robust_subgrad(w_k)
2. Bước xuống theo subgradient: z = w_k - alpha_k * v
3. JOINT PROX của (L1 + chỉ số ràng buộc {1^T w = 1}) tại z, ngưỡng
   t = alpha_k * lam -- MỘT bước duy nhất, xem `_prox_l1_simplex_eq`:
       w_{k+1} = _prox_l1_simplex_eq(z, t)

Bước 2-3 hợp thành một "proximal-subgradient step" chuẩn: subgradient step
cho phần trơn+robust, rồi prox CHÍNH XÁC (không tách rời) của phần không
trơn còn lại (L1 + ràng buộc affine cùng lúc). Xem chứng minh + cách giải
trong "FIX-ROUND: joint prox" bên dưới.

Step size giảm dần: alpha_k = alpha0 / sqrt(k+1) -- lựa chọn kinh điển cho
subgradient method (đảm bảo sum(alpha_k)=inf, sum(alpha_k^2)<inf, điều kiện
đủ cho hội tụ của subgradient method trên hàm lồi).

---------------------------------------------------------------------------
FIX-ROUND: joint prox của lam*||w||_1 + I{1^T w=1} (thay cho prox-then-
project tách rời ở bản đầu)
---------------------------------------------------------------------------
BẢN ĐẦU (đã bị controller/CVXPY cross-check phát hiện lỗi): soft-threshold
L1 rồi CHIẾU RIÊNG lên hyperplane bằng cách cộng offset đều
`(1-sum(w_half))/N` vào MỌI toạ độ. Offset này cộng vào CẢ toạ độ vừa bị
soft-threshold về đúng 0 -- "hồi sinh" toàn bộ toạ độ zero thành một giá trị
nhỏ giống nhau, phá sparsity thật (vd trên data thật: active=48 trong khi
CVXPY active=9 cho kappa=0,gamma=5,lam=0.1; objective gap 2.7-6%). Đây
không phải prox chuẩn của (L1 + ràng buộc affine) -- chỉ là prox L1 rồi
chiếu Euclid tách rời, hai phép toán KHÔNG hoán đổi được với nhau.

BẢN FIX: giải trực tiếp prox CHÍNH XÁC của `lam*||w||_1 + I{1^T w=1}` tại
điểm z (kết quả sau subgradient step), tức:

    w* = argmin_w  (1/2)||w - z||_2^2 + t*||w||_1   s.t.  1^T w = 1,
    với t = alpha_k * lam (thang đo Moreau envelope chuẩn: chọn t = alpha_k
    * lam để phép prox này khớp đúng vai trò của bước 3 cũ trong sơ đồ
    proximal-subgradient, tức vẫn tương đương "soft-threshold ngưỡng
    alpha_k*lam" khi bỏ ràng buộc).

Đưa ràng buộc vào bằng nhân tử Lagrange nu (scalar):
    L(w, nu) = (1/2)||w-z||_2^2 + t*||w||_1 + nu*(1^T w - 1)
Tách theo từng toạ độ i (bài toán tách được hoàn toàn theo i với nu cố
định), tối thiểu hoá:
    (1/2)(w_i - z_i)^2 + nu*w_i + t*|w_i|
Hoàn thiện bình phương: (1/2)(w_i-z_i)^2 + nu*w_i
    = (1/2)(w_i - (z_i - nu))^2 + hằng số (không phụ thuộc w_i)
=> bài toán từng toạ độ trở thành prox L1 chuẩn tại điểm dịch chuyển
(z_i - nu), ngưỡng t:
    w_i(nu) = soft_threshold(z_i - nu, t)
            = sign(z_i - nu) * max(|z_i - nu| - t, 0)

(Quy ước dấu: đặt nu' = -nu, viết lại w_i = soft_threshold(z_i + nu', t) --
đây chính là công thức trong `_prox_l1_simplex_eq`, biến `nu` trong code
tương ứng nu' ở đây; dấu của nu không quan trọng, chỉ là biến cần tìm bằng
bisection.)

Chọn nu (hay nu') sao cho ràng buộc 1^T w = 1 thoả mãn:
    g(nu) := sum_i soft_threshold(z_i + nu, t) = 1

g(nu) là hàm liên tục, KHÔNG GIẢM theo nu (mỗi số hạng soft_threshold(.,t)
không giảm theo đối số của nó, đối số tăng tuyến tính theo nu với hệ số 1)
và không bị chặn cả hai phía (nu -> -inf => g -> -inf, nu -> +inf => g ->
+inf) => tồn tại nu* (có thể không duy nhất nếu g phẳng đúng tại 1, nhưng
mọi nu trong đoạn phẳng đó đều cho nghiệm w hợp lệ) sao cho g(nu*) = 1, tìm
bằng BISECTION:
1. Bracket: bắt đầu [nu_lo, nu_hi] = [-1, 1], nhân đôi nu_lo (nếu g(nu_lo) >
   1) hoặc nu_hi (nếu g(nu_hi) < 1) tới khi bracket đúng hướng.
2. Bisection ~100 vòng hoặc tới khi |g(nu_mid) - 1| < 1e-12.
3. w = soft_threshold(z + nu*, t) -- vector cuối, có toạ độ = 0 CHÍNH XÁC
   với mọi i thoả |z_i + nu*| <= t, và sum(w) = 1 (tới sai số bisection).

Đây là prox CHÍNH XÁC (không phải heuristic) của (L1 + ràng buộc affine)
tại z -- khác biệt so với bản đầu ở chỗ offset không còn cộng đều vào MỌI
toạ độ nữa, mà chỉ dịch chuyển ĐỐI SỐ trước khi soft-threshold, nên toạ độ
đã bị threshold về 0 (|z_i+nu*| <= t) VẪN LÀ 0 sau khi cộng offset -- không
"hồi sinh". Xem `_prox_l1_simplex_eq` để biết cài đặt, và mục "Verify chéo
CVXPY sau fix" trong task-3-report.md cho số liệu thật xác nhận active/gap
cải thiện đáng kể so với bản đầu.

---------------------------------------------------------------------------
Vì sao trả BEST-ITERATE, không phải iterate cuối
---------------------------------------------------------------------------
Subgradient method (khác gradient method trên hàm trơn) KHÔNG đảm bảo
f(w_{k+1}) <= f(w_k) -- dãy f(w_k) có thể dao động không đơn điệu, đặc biệt
ở các vòng đầu khi alpha_k còn lớn. Đây là tính chất TOÁN HỌC cố hữu của
subgradient method, không phải lỗi cài đặt. Do đó ta theo dõi best_obj =
min_{j<=k} f(w_j) và trả best_w tương ứng, thay vì w cuối cùng -- đây là
thực hành chuẩn khi dùng subgradient method (xem Boyd, "Subgradient
Methods" notes).

---------------------------------------------------------------------------
Lựa chọn alpha0 mặc định (TINH CHỈNH LẠI sau FIX joint prox)
---------------------------------------------------------------------------
Daily returns rất nhỏ: mu ~ 1e-3, Sigma ~ 1e-4 -> phần "trơn" của subgradient
(-mu + 2*gamma*Sigma@w) có magnitude ~ 1e-3..1e-2 tuỳ gamma, trong khi robust
term kappa*Sigma@w/||.|| có magnitude ~ kappa (order 1 vì đã chia norm).

Với bản prox-then-project CŨ, alpha0 lớn làm sparsity bị offset "hồi sinh"
(xem lịch sử trong mục "FIX-ROUND" ở trên) -- nên trước đây phải chọn alpha0
NHỎ (0.1) để giữ sparsity, đánh đổi hội tụ chậm/best_obj kém hơn. Với joint
prox (bản FIX), vấn đề đó KHÔNG còn: toạ độ bị threshold về 0 vẫn là 0 dù
alpha0 lớn, nên KHÔNG còn đánh đổi giữa sparsity và tốc độ hội tụ -- alpha0
lớn hơn vừa hội tụ nhanh hơn (patience trigger sớm), vừa best_obj thấp hơn
(hoặc bằng), vừa sparsity bằng hoặc TỐT hơn.

Thử nghiệm thật trên data/returns.parquet (97 tài sản, max_iter=20000, xem
task-3-report.md mục "Fix-round: tinh chỉnh lại alpha0") với alpha0 in
{0.01, 0.05, 0.1, 0.3, 1, 3, 10, 30, 100, 300} cho ba bộ tham số
(kappa=1,gamma=5,lam=0.01), (kappa=0,gamma=5,lam=0.1),
(kappa=1,gamma=5,lam=0.001): best_obj và active/sparsity đều CẢI THIỆN đơn
điệu (hoặc bão hoà) khi alpha0 tăng từ 0.01 -> 10, sau đó ổn định (alpha0=10,
30 cho kết quả gần như giống hệt alpha0=1..3 nhưng hội tụ nhanh hơn nhiều
lần: vài trăm tới vài nghìn vòng thay vì hết max_iter). Ở alpha0=100 (bộ
kappa=1,gamma=5,lam=0.001), bước đầu quá dài gây dao động mạnh khiến
best-iterate mắc kẹt sớm tại điểm KÉM (best_obj tệ hơn hẳn, active=97) --
"converged" giả giống hiện tượng đã thấy ở bản cũ. Do đó cần alpha0 đủ lớn
để hội tụ nhanh nhưng chưa chạm vùng mất ổn định đó.

ĐƯỢC CHỌN: ALPHA0_DEFAULT = 10.0 -- nằm giữa vùng ổn định (10, 30 cho kết
quả gần như tối ưu quan sát được trong grid, trong khi 100 đã mất ổn định ở
một bộ tham số) với biên an toàn 3x trước ngưỡng mất ổn định gần nhất quan
sát được (30 -> 100). Caller vẫn có thể truyền alpha0 khác nếu cần (vd bộ
tham số khác biệt scale nhiều so với daily returns VN100). Số liệu đầy đủ
xem task-3-report.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SolveResult",
    "portfolio_objective",
    "solve",
    "simplex_projection",
    "portfolio_objective_long_only",
    "solve_long_only",
]

# Ngưỡng an toàn chia-cho-0 khi tính subgradient của ||Sigma^(1/2) w||_2.
_EPS_NORM = 1e-12

# Xem "Lựa chọn alpha0 mặc định" ở docstring module.
ALPHA0_DEFAULT = 10.0


def portfolio_objective(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    lam: float,
) -> float:
    """f(w) = -mu^T w + kappa*||Sigma^(1/2) w||_2 + gamma*w^T Sigma w + lam*||w||_1.

    Trả giá trị đầy đủ của hàm mục tiêu (KHÔNG gồm ràng buộc 1^T w = 1 --
    caller tự đảm bảo w feasible nếu muốn giá trị có ý nghĩa cho bài toán
    ràng buộc; hàm này chỉ tính f(w) tại w bất kỳ, kể cả w không feasible,
    hữu ích cho Phase 4 khi CVXPY cần so sánh giá trị mục tiêu tại nghiệm
    của nó).

    Parameters
    ----------
    w : np.ndarray, shape (N,)
    mu : np.ndarray, shape (N,)
    sigma : np.ndarray, shape (N,N)
    sigma_sqrt : np.ndarray, shape (N,N)  -- căn bậc hai đối xứng PSD của sigma
    kappa, gamma, lam : float, hệ số không âm

    Returns
    -------
    float
    """
    w = np.asarray(w, dtype=np.float64)
    mean_term = -float(mu @ w)
    robust_term = kappa * float(np.linalg.norm(sigma_sqrt @ w))
    var_term = gamma * float(w @ sigma @ w)
    l1_term = lam * float(np.sum(np.abs(w)))
    return mean_term + robust_term + var_term + l1_term


def _robust_subgrad(
    w: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    eps: float = _EPS_NORM,
) -> np.ndarray:
    """Subgradient của kappa*||Sigma^(1/2) w||_2 theo w.

    = kappa * (Sigma @ w) / ||Sigma^(1/2) w||_2   nếu ||Sigma^(1/2) w||_2 > eps
    = 0 (vector)                                   nếu ||Sigma^(1/2) w||_2 <= eps

    LƯU Ý: tử số là Sigma @ w (ma trận Sigma đầy đủ), KHÔNG PHẢI
    Sigma_sqrt @ w -- xem giải thích chain-rule ở docstring module. Nhánh
    eps tránh chia cho 0 / NaN khi w gần 0 theo chuẩn Sigma^(1/2) (chọn 0,
    một phần tử hợp lệ của tập dưới vi phân norm tại điểm kỳ dị).
    """
    u = sigma_sqrt @ w
    norm_u = float(np.linalg.norm(u))
    if norm_u <= eps:
        return np.zeros_like(w)
    return kappa * (sigma @ w) / norm_u


def _prox_l1_simplex_eq(
    z: np.ndarray,
    t: float,
    nu_init: float = 1.0,
    bisect_tol: float = 1e-12,
    max_bracket_expand: int = 200,
    max_bisect_iter: int = 200,
) -> np.ndarray:
    """Prox CHÍNH XÁC của `t*||w||_1 + I{1^T w = 1}` tại điểm z.

    Giải:
        w* = argmin_w  (1/2)||w-z||_2^2 + t*||w||_1   s.t.  1^T w = 1

    Đạo hàm đầy đủ (Lagrangian, tách theo toạ độ) nằm ở docstring module,
    mục "FIX-ROUND: joint prox". Tóm tắt: với nu là nhân tử Lagrange (dấu đã
    quy ước để công thức là +nu, xem docstring module),

        w_i(nu) = soft_threshold(z_i + nu, t) = sign(z_i+nu)*max(|z_i+nu|-t, 0)

    và ta cần tìm nu sao cho g(nu) := sum_i w_i(nu) = 1. g liên tục, KHÔNG
    GIẢM theo nu (mỗi số hạng không giảm theo đối số của nó), không bị chặn
    hai phía -- luôn tồn tại nghiệm, tìm bằng BISECTION (không có công thức
    đóng vì soft_threshold không tuyến tính toàn cục do đoạn phẳng quanh 0).

    Kết quả: toạ độ i có |z_i + nu*| <= t sẽ là 0 CHÍNH XÁC (không bị "hồi
    sinh" bởi phần chiếu ràng buộc như bản heuristic prox-then-project cũ
    -- xem CAVEAT lịch sử trong docstring module), và sum(w) = 1 tới sai số
    `bisect_tol`.

    Parameters
    ----------
    z : np.ndarray, shape (N,)
        Điểm cần lấy prox (kết quả sau subgradient step: z = w_k - alpha_k*v).
    t : float, >= 0
        Ngưỡng soft-threshold (= alpha_k * lam trong `solve`). t=0 ->
        hàm rút gọn về đúng phép chiếu Euclid lên hyperplane {1^T w=1}
        (không có soft-threshold), vì soft_threshold(x, 0) = x.
    nu_init : float, default 1.0
        Biên khởi tạo bracket [-nu_init, nu_init] trước khi mở rộng.
    bisect_tol : float, default 1e-12
        Ngưỡng |g(nu)-1| để dừng bisection.
    max_bracket_expand, max_bisect_iter : int
        Giới hạn an toàn số vòng mở rộng bracket / bisection (tránh vòng lặp
        vô hạn trong trường hợp số học biên; về lý thuyết luôn hội tụ).

    Returns
    -------
    np.ndarray, shape (N,), thoả sum(w) ~= 1 (tới bisect_tol) và có thể có
    toạ độ bằng 0 chính xác.
    """
    z = np.asarray(z, dtype=np.float64)

    def g(nu: float) -> float:
        x = z + nu
        return float(np.sum(np.sign(x) * np.maximum(np.abs(x) - t, 0.0)))

    nu_lo, nu_hi = -nu_init, nu_init
    expand = 0
    while g(nu_lo) > 1.0 and expand < max_bracket_expand:
        nu_lo *= 2.0
        expand += 1
    expand = 0
    while g(nu_hi) < 1.0 and expand < max_bracket_expand:
        nu_hi *= 2.0
        expand += 1

    nu_mid = 0.5 * (nu_lo + nu_hi)
    for _ in range(max_bisect_iter):
        nu_mid = 0.5 * (nu_lo + nu_hi)
        g_mid = g(nu_mid)
        if abs(g_mid - 1.0) < bisect_tol:
            break
        if g_mid < 1.0:
            nu_lo = nu_mid
        else:
            nu_hi = nu_mid

    x = z + nu_mid
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _smooth_subgrad(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    eps: float = _EPS_NORM,
) -> np.ndarray:
    """v = -mu + 2*gamma*Sigma@w + robust_subgrad(w) (KHÔNG gồm phần L1,
    L1 xử lý riêng bằng joint prox `_prox_l1_simplex_eq` trong `solve`)."""
    return -mu + 2.0 * gamma * (sigma @ w) + _robust_subgrad(w, sigma, sigma_sqrt, kappa, eps)


@dataclass
class SolveResult:
    """Kết quả trả về bởi `solve`.

    w : np.ndarray, shape (N,)
        Nghiệm BEST-ITERATE (KHÔNG phải iterate cuối -- xem docstring module
        "Vì sao trả best-iterate").
    obj_history : np.ndarray, shape (n_iter,)
        f(w_k) tại MỖI vòng lặp thực sự chạy (w_k đã feasible vì đã chiếu
        hyperplane), theo thứ tự thời gian -- KHÔNG phải running-min.
    best_obj : float
        min(obj_history) = f(w) tại w trả về.
    n_iter : int
        Số vòng lặp thực sự đã chạy (<= max_iter).
    converged : bool
        True nếu dừng do relative-change của best_obj < tol trong `patience`
        vòng liên tiếp; False nếu dừng do chạm max_iter.
    """

    w: np.ndarray
    obj_history: np.ndarray
    best_obj: float
    n_iter: int
    converged: bool


def solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    lam: float,
    *,
    max_iter: int = 5000,
    alpha0: float = ALPHA0_DEFAULT,
    tol: float = 1e-8,
    patience: int = 100,
    w0: np.ndarray | None = None,
) -> SolveResult:
    """Giải min_w f(w) s.t. 1^T w = 1 bằng proximal-subgradient tự viết.

    Xem docstring module cho công thức thuật toán đầy đủ, đạo hàm robust
    term, lý do trả best-iterate, và mục "FIX-ROUND: joint prox" (bước
    L1 + ràng buộc affine giải CHÍNH XÁC bằng bisection, không còn heuristic
    prox-then-project của bản đầu).

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
        Xem `portfolio_objective`.
    kappa, gamma, lam : float
        Hệ số không âm của robust / variance / L1 term.
    max_iter : int, default 5000
    alpha0 : float, default ALPHA0_DEFAULT (=10.0, xem docstring module)
        alpha_k = alpha0 / sqrt(k+1).
    tol : float, default 1e-8
        Ngưỡng relative-change của best_obj để coi là "đã ổn định".
    patience : int, default 100
        Số vòng liên tiếp relative-change < tol để dừng sớm (converged=True).
    w0 : np.ndarray | None, default None
        Điểm khởi tạo; None -> vector đều 1/N (feasible: sum=1).

    Returns
    -------
    SolveResult
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_sqrt = np.asarray(sigma_sqrt, dtype=np.float64)
    n = mu.shape[0]

    if w0 is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(w0, dtype=np.float64).copy()

    obj_hist = np.empty(max_iter, dtype=np.float64)

    best_obj = portfolio_objective(w, mu, sigma, sigma_sqrt, kappa, gamma, lam)
    best_w = w.copy()
    prev_best = best_obj
    stall_count = 0
    converged = False
    n_iter = 0

    for k in range(max_iter):
        alpha_k = alpha0 / np.sqrt(k + 1)

        v = _smooth_subgrad(w, mu, sigma, sigma_sqrt, kappa, gamma)
        z = w - alpha_k * v

        # Joint prox CHÍNH XÁC của (lam*||.||_1 + I{1^T w=1}) tại z, ngưỡng
        # alpha_k*lam -- xem docstring module "FIX-ROUND: joint prox" và
        # `_prox_l1_simplex_eq`. Thay thế bản cũ (soft-threshold rồi chiếu
        # tách rời) vốn phá sparsity do offset cộng đều vào mọi toạ độ.
        w = _prox_l1_simplex_eq(z, alpha_k * lam)

        f_w = portfolio_objective(w, mu, sigma, sigma_sqrt, kappa, gamma, lam)
        obj_hist[k] = f_w
        n_iter = k + 1

        if f_w < best_obj:
            best_obj = f_w
            best_w = w.copy()

        denom = max(abs(prev_best), 1e-300)
        rel_change = abs(prev_best - best_obj) / denom
        prev_best = best_obj

        if rel_change < tol:
            stall_count += 1
        else:
            stall_count = 0

        if stall_count >= patience:
            converged = True
            break

    return SolveResult(
        w=best_w,
        obj_history=obj_hist[:n_iter],
        best_obj=best_obj,
        n_iter=n_iter,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Walk-forward Task 1: nhánh long-only (w >= 0, sum(w) = 1)
# ---------------------------------------------------------------------------
# Dưới ràng buộc long-only, ||w||_1 = sum_i |w_i| = sum_i w_i = 1 (vì mọi
# w_i >= 0) LÀ HẰNG SỐ trên toàn miền khả thi -- phạt lam*||w||_1 không còn
# tác dụng điều khiển nghiệm (nó chỉ cộng thêm đúng "lam" vào objective, một
# hằng số không phụ thuộc w) nên bị loại khỏi nhánh này hoàn toàn (không có
# tham số lam trong `portfolio_objective_long_only` / `solve_long_only`).
# Xem docs/superpowers/specs/2026-07-26-walk-forward-backtest-design.md mục 2.
#
# Phần "trơn + robust" của subgradient (-mu + 2*gamma*Sigma@w +
# robust_subgrad(w)) giữ NGUYÊN công thức như nhánh long-short ở trên -- tái
# dùng trực tiếp `_smooth_subgrad` (không sửa, không copy lại công thức) để
# tránh lệch pha với `solve()` nếu công thức đó có thay đổi sau này. Điểm
# khác biệt DUY NHẤT so với `solve()`: bước "prox" của (L1 + ràng buộc affine)
# được thay bằng phép CHIẾU EUCLID lên simplex `{w>=0, sum(w)=1}`
# (`simplex_projection`, thuật toán Duchi et al. 2008) -- vì không còn L1
# nên không cần joint-prox qua bisection như `_prox_l1_simplex_eq`.
# ---------------------------------------------------------------------------


def simplex_projection(v: np.ndarray) -> np.ndarray:
    """Chiếu Euclid vector v lên probability simplex {w : w>=0, sum(w)=1}.

    Thuật toán Duchi, Shalev-Shwartz, Singer, Chandra (2008) "Efficient
    Projections onto the l1-Ball for Learning in High Dimensions", O(N log N):
    sort v giảm dần thành u; tìm rho = chỉ số lớn nhất j sao cho
    u_j - (cumsum(u)_j - 1)/j > 0; theta = (cumsum(u)_rho - 1)/rho;
    w = max(v - theta, 0). Kết quả LUÔN thoả w>=0 và sum(w)=1 (tới sai số
    số học), bất kể v là gì -- không cần v đã "gần" simplex.

    Parameters
    ----------
    v : np.ndarray, shape (N,)

    Returns
    -------
    np.ndarray, shape (N,), thoả w>=0 (tới sai số số học) và sum(w)=1.
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cumsum_u = np.cumsum(u)
    j = np.arange(1, n + 1)
    cond = u - (cumsum_u - 1) / j > 0
    rho = np.nonzero(cond)[0][-1]  # chỉ số 0-based của j lớn nhất thoả cond
    theta = (cumsum_u[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def portfolio_objective_long_only(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
) -> float:
    """f(w) = -mu^T w + kappa*||Sigma^(1/2) w||_2 + gamma*w^T Sigma w.

    Giống `portfolio_objective` nhưng KHÔNG có term `lam*||w||_1` -- dưới
    ràng buộc long-only (w>=0, sum(w)=1) thì ||w||_1 = sum(w) = 1 là hằng số,
    phạt L1 không còn tác dụng điều khiển nghiệm nên bị loại (xem docstring
    ở khối "Walk-forward Task 1" phía trên).

    Parameters
    ----------
    w, mu, sigma, sigma_sqrt : np.ndarray -- xem `portfolio_objective`.
    kappa, gamma : float, hệ số không âm.

    Returns
    -------
    float
    """
    w = np.asarray(w, dtype=np.float64)
    mean_term = -float(mu @ w)
    robust_term = kappa * float(np.linalg.norm(sigma_sqrt @ w))
    var_term = gamma * float(w @ sigma @ w)
    return mean_term + robust_term + var_term


def solve_long_only(
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_sqrt: np.ndarray,
    kappa: float,
    gamma: float,
    *,
    max_iter: int = 5000,
    alpha0: float = ALPHA0_DEFAULT,
    tol: float = 1e-8,
    patience: int = 100,
    w0: np.ndarray | None = None,
) -> SolveResult:
    """Giải min_w f(w) s.t. w>=0, sum(w)=1 bằng projected-subgradient tự viết.

    Cùng sơ đồ vòng lặp như `solve()` (subgradient step trên phần trơn+robust
    qua `_smooth_subgrad`, KHÔNG sửa/copy lại công thức, rồi bước "prox"),
    chỉ khác bước prox: ở đây không còn term L1 (xem khối docstring
    "Walk-forward Task 1" phía trên) nên bước prox rút gọn thành phép chiếu
    Euclid lên simplex qua `simplex_projection` (Duchi et al. 2008), thay cho
    `_prox_l1_simplex_eq` của nhánh long-short. Cùng lý do trả BEST-ITERATE
    (subgradient method không đảm bảo đơn điệu) như `solve()` -- xem docstring
    module mục "Vì sao trả BEST-ITERATE".

    Parameters
    ----------
    mu, sigma, sigma_sqrt : np.ndarray
        Xem `portfolio_objective`.
    kappa, gamma : float, hệ số không âm.
    max_iter : int, default 5000
    alpha0 : float, default ALPHA0_DEFAULT
        alpha_k = alpha0 / sqrt(k+1).
    tol : float, default 1e-8
        Ngưỡng relative-change của best_obj để coi là "đã ổn định".
    patience : int, default 100
        Số vòng liên tiếp relative-change < tol để dừng sớm (converged=True).
    w0 : np.ndarray | None, default None
        Điểm khởi tạo; None -> vector đều 1/N (feasible: w>=0, sum=1).

    Returns
    -------
    SolveResult
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_sqrt = np.asarray(sigma_sqrt, dtype=np.float64)
    n = mu.shape[0]

    if w0 is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(w0, dtype=np.float64).copy()

    obj_hist = np.empty(max_iter, dtype=np.float64)

    best_obj = portfolio_objective_long_only(w, mu, sigma, sigma_sqrt, kappa, gamma)
    best_w = w.copy()
    prev_best = best_obj
    stall_count = 0
    converged = False
    n_iter = 0

    for k in range(max_iter):
        alpha_k = alpha0 / np.sqrt(k + 1)

        v = _smooth_subgrad(w, mu, sigma, sigma_sqrt, kappa, gamma)
        z = w - alpha_k * v

        # Chiếu Euclid lên simplex {w>=0, sum(w)=1} -- thay cho joint prox
        # (L1 + hyperplane) của solve(), vì nhánh long-only không có L1.
        w = simplex_projection(z)

        f_w = portfolio_objective_long_only(w, mu, sigma, sigma_sqrt, kappa, gamma)
        obj_hist[k] = f_w
        n_iter = k + 1

        if f_w < best_obj:
            best_obj = f_w
            best_w = w.copy()

        denom = max(abs(prev_best), 1e-300)
        rel_change = abs(prev_best - best_obj) / denom
        prev_best = best_obj

        if rel_change < tol:
            stall_count += 1
        else:
            stall_count = 0

        if stall_count >= patience:
            converged = True
            break

    return SolveResult(
        w=best_w,
        obj_history=obj_hist[:n_iter],
        best_obj=best_obj,
        n_iter=n_iter,
        converged=converged,
    )
