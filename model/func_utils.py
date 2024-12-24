import numpy as np
from scipy.special import gammaln, digamma
from scipy.stats import linregress
import matplotlib.pyplot as plt

# 传入的参数已经是对次数的统计值了
def DS_Combine_ensemble_for_instances(E1, E2):
    n_classes = E1.shape[1]
    alpha1 = E1 + 1
    alpha2 = E2 + 1

    S1 = np.sum(alpha1, axis=1, keepdims=True)
    S2 = np.sum(alpha2, axis=1, keepdims=True)

    print("S1", S1)
    print("S2", S2)
    b1 = E1 / S1
    b2 = E2 / S2

    # print("b1", b1)
    # print("b2", b2)

    u1 = n_classes / S1
    u2 = n_classes / S2

    # print("u1", u1)
    # print("u2", u2)

    bb = np.einsum('ij,ik->ijk', b1, b2)

    # 计算 b^0 * u^1
    uv1_expand = np.broadcast_to(u2, b1.shape)  # 使用 np.broadcast_to 匹配形状
    bu = b1 * uv1_expand

    # 计算 b^1 * u^0
    uv_expand = np.broadcast_to(u1, b2.shape)
    ub = b2 * uv_expand

    # 计算 K
    bb_sum = np.sum(bb, axis=(1, 2))  # 计算 bb 的总和
    bb_diag = np.einsum('ijj->i', bb)  # 提取对角线并在批量中求和
    K = bb_sum - bb_diag

    # 计算 b^a
    b_combined = (b1 * b2 + bu + ub) / np.expand_dims((1 - K), axis=1)

    # 计算 u^a
    u_combined = (u1 * u2) / np.expand_dims((1 - K), axis=1)

    # 计算新的 S
    S_combined = n_classes / u_combined

    # 计算新的 e_k
    e_combined = b_combined * np.broadcast_to(S_combined, b_combined.shape)
    alpha_combined = e_combined + 1

    return alpha_combined, b_combined, u_combined




# 需要传入的参数：
# alpha1 = evidence1 + 1
# alpha2 = evidence2 + 1
def DS_Combin_two(alpha1, alpha2, n_classes):
    # 计算两个DS证据的合并
    alpha = {0: alpha1, 1: alpha2}
    b, S, E, u = {}, {}, {}, {}

    for v in range(2):

        # S[v]将每个样本的所有类别的alpha[v]相加，每个样本的S值都为 4
        S[v] = np.sum(alpha[v], axis=1, keepdims=True)# 使用 np.sum 计算 S
        E[v] = alpha[v] - 1
        # print("alpha[v].shape", alpha[v].shape)
        # print("E[v].shape", E[v].shape)
        # print("S[v].shape", S[v].shape)
        # print("alpha[v]", alpha[v])
        # print("E[v]", E[v])
        # print("S[v]", S[v])
        # b[v] = E[v] / np.expand_dims(S[v], axis=1)  # 使用 np.expand_dims 进行广播
        b[v] = E[v] / S[v]
        # print("np.expand_dims(S[v], axis=1).shape", np.expand_dims(S[v], axis=1).shape)
        # print("b[v].shape", b[v].shape)
        u[v] = n_classes / S[v]

        print(u[v])

    # 计算 b^0 @ b^(0+1)

    # print("b[0].shape", b[0].shape)
    # print("b[1].shape", b[1].shape)
    bb = np.einsum('ij,ik->ijk', b[0], b[1])  # 使用 np.einsum 实现批量矩阵乘法
    # print("bb.shape", bb.shape)



    # 计算 b^0 * u^1
    uv1_expand = np.broadcast_to(u[1], b[0].shape)  # 使用 np.broadcast_to 匹配形状
    bu = b[0] * uv1_expand

    # 计算 b^1 * u^0
    uv_expand = np.broadcast_to(u[0], b[1].shape)
    ub = b[1] * uv_expand

    # 计算 K
    bb_sum = np.sum(bb, axis=(1, 2))  # 计算 bb 的总和
    bb_diag = np.einsum('ijj->i', bb)  # 提取对角线并在批量中求和
    K = bb_sum - bb_diag

    # 计算 b^a
    b_combined = (b[0] * b[1] + bu + ub) / np.expand_dims((1 - K), axis=1)

    # 计算 u^a
    u_combined = (u[0] * u[1]) / np.expand_dims((1 - K), axis=1)

    # 计算新的 S
    S_combined = n_classes / u_combined

    # 计算新的 e_k
    e_combined = b_combined * np.broadcast_to(S_combined, b_combined.shape)
    alpha_combined = e_combined + 1

    return alpha_combined, b_combined, u_combined


# 需要传入的参数：
# alpha = evidence + 1
# c = n_classes
def KL(alpha, c):
    beta = np.ones((1, c))
    S_alpha = np.sum(alpha, axis=1, keepdims=True)  # 使用 np.sum 计算 S_alpha
    S_beta = np.sum(beta, axis=1, keepdims=True)

    # 使用 scipy.special 中的 gammaln 计算 lnB 和 lnB_uni
    lnB = gammaln(S_alpha) - np.sum(gammaln(alpha), axis=1, keepdims=True)
    lnB_uni = np.sum(gammaln(beta), axis=1, keepdims=True) - gammaln(S_beta)

    # 计算 digamma 值
    dg0 = digamma(S_alpha)
    dg1 = digamma(alpha)

    # 计算 KL 散度
    # print("(alpha - beta).shape", (alpha - beta).shape)
    # print("(dg1 - dg0).shape", (dg1 - dg0).shape)
    kl = np.sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB + lnB_uni
    return kl

def calculate_KL(alpha, c):
    return KL(alpha, c)

def calculate_A(alpha, p, c):
    S = np.sum(alpha, axis=1, keepdims=True)
    label = np.eye(c)[p]  # 创建独热编码标签
    A = np.sum(label * (digamma(S) - digamma(alpha)), axis=1, keepdims=True)
    return A

# 需要传入的参数：
# p 样本真实标签
# alpha = evidence + 1
# c = n_classes
# global_step 当前训练步数
# annealing_step 退火步数
def ce_loss(p, alpha, c, global_step=0, annealing_step=0, average=True):
    # 计算 S 和 E
    S = np.sum(alpha, axis=1, keepdims=True)
    E = alpha - 1

    # print("S:\n", S)
    # print("E:\n", E)
    # 独热编码
    label = np.eye(c)[p]  # 创建独热编码标签
    # print("label", label)
    # 计算 A 项
    A = np.sum(label * (digamma(S) - digamma(alpha)), axis=1, keepdims=True)

    # 计算退火系数
    # annealing_coef = min(1, global_step / annealing_step)

    # 先让系数固定
    annealing_coef = 1

    # print("1 - label:\n", 1 - label)
    # 计算 alp 和 B 项
    alp = E * (1 - label) + 1
    # print("alp:\n", alp)

    # 现在B项是随着alpha的分布越接近均匀分布，KL散度越大
    B = annealing_coef * KL(alp, c)

    # print("A:\n", A)
    # print("B:\n", B)
    # print("A.shape:\n", A.shape)
    # print("B.shape:\n", B.shape)
    # 返回 (A + B) 的均值
    if average is True:
        res = np.mean(A + B)
        return res, A.reshape(-1, 1), B.reshape(-1, 1)
    else:
        # print("res:\n", A + B)
        res = A + B

        # 用一个y = kx + b的图像来画出A和B的变化
        A = A.flatten()
        B = B.flatten()

        # 绘制 A 和 B 的散点图
        num_bins = 50  # 可以根据需求调整区间数量

        # 计算 A 的分布区间，并对频数取对数
        A_counts, A_bins = np.histogram(A, bins=num_bins)
        A_counts_log = np.log(A_counts + 1e-6)  # 对频数取对数，避免 log(0)

        # 绘制 A 的正常频数分布直方图
        # plt.figure(figsize=(8, 5))
        # plt.bar(A_bins[:-1], A_counts, width=(A_bins[1] - A_bins[0]), color="blue", alpha=0.7, edgecolor="black")
        # plt.title("Distribution of Dirichlet Expected Cross-Entropy Frequency")
        # plt.xlabel("A Value")
        # plt.ylabel("Frequency")
        # plt.show()
        #
        # # 绘制 A 的对数化频数分布直方图
        # plt.figure(figsize=(8, 5))
        # plt.bar(A_bins[:-1], A_counts_log, width=(A_bins[1] - A_bins[0]), color="blue", alpha=0.7, edgecolor="black")
        # plt.title("Logarithmic Distribution of Dirichlet Expected Cross-Entropy Frequency")
        # plt.xlabel("A Value")
        # plt.ylabel("Log(Frequency)")
        # plt.show()

        # 计算 B 的分布区间，并对频数取对数
        B_counts, B_bins = np.histogram(B, bins=num_bins)
        B_counts_log = np.log(B_counts + 1e-6)  # 对频数取对数，避免 log(0)
        # print("\nB 的区间分布:")
        # for i in range(len(B_bins) - 1):
        #     print(f"区间 {B_bins[i]:.2f} - {B_bins[i + 1]:.2f}: {B_counts[i]} 个样本")

        # 绘制 B 的正常频数分布直方图
        # plt.figure(figsize=(8, 5))
        # plt.bar(B_bins[:-1], B_counts, width=(B_bins[1] - B_bins[0]), color="orange", alpha=0.7, edgecolor="black")
        # plt.title("Distribution of KL-Divergence Frequency")
        # plt.xlabel("B Value")
        # plt.ylabel("Frequency")
        # plt.show()
        #
        # # 绘制 B 的对数化频数分布直方图
        # plt.figure(figsize=(8, 5))
        # plt.bar(B_bins[:-1], B_counts_log, width=(B_bins[1] - B_bins[0]), color="orange", alpha=0.7, edgecolor="black")
        # plt.title("Logarithmic Distribution of KL-Divergence Frequency")
        # plt.xlabel("B Value")
        # plt.ylabel("Log(Frequency)")
        # plt.show()


        # 执行线性回归，找出 k 和 b
        slope, intercept, r_value, p_value, std_err = linregress(A, B)

        # 生成拟合直线的 y 值
        B_fit = slope * A + intercept

        # 绘制散点图和拟合直线
        # plt.figure(figsize=(10, 5))
        # plt.scatter(A, B, label="Data", alpha=0.3, color="blue")
        # plt.plot(A, B_fit, color="red", label=f"Fit: B = {slope:.2f}A + {intercept:.2f}")
        # plt.xlabel("A")
        # plt.ylabel("B")
        # plt.title("Relationship between A and B with Linear Fit")
        # plt.legend()
        # plt.show()

        # 输出拟合的斜率和截距
        # print(f"线性拟合结果：B = {slope:.2f}A + {intercept:.2f}")
        # print(f"相关系数 (R^2): {r_value ** 2:.2f}")

        return res, A.reshape(-1, 1), B.reshape(-1, 1)
