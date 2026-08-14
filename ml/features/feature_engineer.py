"""
双色球预测系统 - 特征工程模块
整合所有特征计算逻辑，提供统一的特征工程接口。

支持的特征类型:
- 统计特征 (sum, mean, std, skew, kurtosis, min, max, range)
- 频率特征 (每个号码的出现次数)
- 区间特征 (号码区间分布)
- 奇偶特征
- 大小特征 (以17为界)
- 连号特征
- 质数特征
- 位置特征
- 近期频率特征 (滑动窗口)
- 遗漏特征 (上次出现位置)
- 泊松分布特征
- 正态分布特征
- 信息熵特征
- 马尔可夫链特征
- 冷热号特征
- 正弦余弦编码
- 大数定律特征
- 和值区间特征
- 区间分布特征
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, poisson
from typing import List, Optional, Dict, Any

from ml.config import (
    FEATURE_CONFIG,
    RED_COLS,
    BLUE_COLS,
    TARGET_COLS,
    RED_RANGE,
    RED_NUMBERS,
    BLUE_RANGE,
    BLUE_NUMBERS,
)


def markov_chi_square_test(series: np.ndarray, states: int = 7, alpha: float = 0.05):
    """卡方马氏性检验: 序列是否可用一阶马尔可夫链建模。

    原理解释(阿里云 264900 方法论): 若序列是纯随机(无记忆), 则状态 i 的
    下一个状态分布应与"全样本状态频率"一致。构造列联表:
      行 = 当前状态 i, 列 = 下一状态 j, 单元格 = 实际转移计数 n_ij;
      期望单元格 = row_total_i * col_total_j / grand_total。
    卡方统计量 = Σ (n_ij - E_ij)^2 / E_ij, 自由度 = (states-1)^2。
    p < alpha -> 拒绝"独立/无记忆"原假设 -> 序列具马氏性(可用马尔可夫)。
    p >= alpha -> 不拒绝原假设 -> 序列近似无记忆 -> 用马尔可夫预测无意义。

    对双色球这类强随机序列, 预期 p 值偏高(不通过), 正好用数据说话。

    Args:
        series: 离散状态序列(如每期奇偶比 0..6)。
        states: 状态数。
        alpha: 显著性水平。

    Returns:
        (chi2_stat, p_value, is_markov) 元组。is_markov=True 表示通过检验
        (存在马氏性, 马尔可夫特征可用)。
    """
    from scipy.stats import chi2

    seq = np.asarray(series, dtype=int)
    seq = seq[~np.isnan(seq)]
    if len(seq) < states * 2:
        # 样本不足, 无法可靠检验 -> 保守判定无效
        return 0.0, 1.0, False

    # 列联表 n_ij: 从 i 转移到 j 的计数
    n = np.zeros((states, states), dtype=float)
    for i in range(1, len(seq)):
        a, b = seq[i - 1], seq[i]
        if 0 <= a < states and 0 <= b < states:
            n[a, b] += 1

    row_tot = n.sum(axis=1)
    col_tot = n.sum(axis=0)
    grand = n.sum()
    if grand == 0 or np.any(row_tot == 0) or np.any(col_tot == 0):
        return 0.0, 1.0, False

    # 期望计数(独立假设下)
    e = np.outer(row_tot, col_tot) / grand
    # 仅对 E_ij > 0 的单元格累加(避免除零)
    mask = e > 0
    chi2_stat = float(np.sum((n[mask] - e[mask]) ** 2 / e[mask]))
    dof = (states - 1) ** 2
    p_value = float(chi2.sf(chi2_stat, dof))
    is_markov = bool(p_value < alpha)
    return chi2_stat, p_value, is_markov


class FeatureEngineer:
    """特征工程统一接口

    整合分散在各脚本中的特征计算逻辑，提供可复用的特征计算方法。
    所有方法均以DataFrame为输入，返回添加了特征列的DataFrame。

    Attributes:
        red_cols: 红球列名列表
        blue_cols: 蓝球列名列表
        target_cols: 所有目标列名列表
        red_numbers: 红球号码数量 (33)
        blue_numbers: 蓝球号码数量 (16)
        primes: 质数集合
    """

    def __init__(
        self,
        red_cols: Optional[List[str]] = None,
        blue_cols: Optional[List[str]] = None,
        primes: Optional[set] = None,
    ):
        """初始化特征工程器

        Args:
            red_cols: 红球列名列表，默认使用配置
            blue_cols: 蓝球列名列表，默认使用配置
            primes: 质数集合，默认使用配置
        """
        self.red_cols = red_cols or RED_COLS
        self.blue_cols = blue_cols or BLUE_COLS
        self.target_cols = self.red_cols + self.blue_cols
        self.red_numbers = RED_NUMBERS
        self.blue_numbers = BLUE_NUMBERS
        self.primes = primes or FEATURE_CONFIG["primes"]

    def calc_statistical_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """统计特征：和值、均值、标准差、偏度、峰度、最小值、最大值、极差

        Args:
            window_data: 窗口数据DataFrame，每行一期开奖数据

        Returns:
            包含统计特征的DataFrame
        """
        features: Dict[str, np.ndarray] = {}
        row_values = window_data.values

        features["Sum"] = row_values.sum(axis=1)
        features["Mean"] = row_values.mean(axis=1)
        features["Std"] = row_values.std(axis=1)
        features["Min"] = row_values.min(axis=1)
        features["Max"] = row_values.max(axis=1)
        features["Range"] = features["Max"] - features["Min"]
        features["Skew"] = skew(row_values, axis=1)
        features["Kurtosis"] = kurtosis(row_values, axis=1)

        return pd.DataFrame(features)

    def calc_frequency_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """频率特征：每个数字(1-33)在窗口中的出现次数，及唯一值计数

        Args:
            window_data: 窗口数据DataFrame

        Returns:
            包含频率特征的DataFrame
        """
        features: Dict[str, np.ndarray] = {}
        row_values = window_data.values

        for num in range(1, self.red_numbers + 1):
            features[f"Freq_{num}"] = (row_values == num).sum(axis=1)

        features["Unique_Count"] = pd.DataFrame(row_values).nunique(axis=1).values

        return pd.DataFrame(features)

    def calc_interval_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """区间特征：各区间(1-11, 12-22, 23-33)数字数量及区间极值

        Args:
            window_data: 窗口数据DataFrame

        Returns:
            包含区间特征的DataFrame
        """
        features: Dict[str, np.ndarray] = {}
        row_values = window_data.values

        features["Int_1_11"] = ((row_values >= 1) & (row_values <= 11)).sum(axis=1)
        features["Int_12_22"] = ((row_values >= 12) & (row_values <= 22)).sum(axis=1)
        features["Int_23_33"] = ((row_values >= 23) & (row_values <= 33)).sum(axis=1)

        features["Int_Max"] = np.maximum.reduce(
            [features["Int_1_11"], features["Int_12_22"], features["Int_23_33"]]
        )
        features["Int_Min"] = np.minimum.reduce(
            [features["Int_1_11"], features["Int_12_22"], features["Int_23_33"]]
        )

        return pd.DataFrame(features)

    def calc_odd_even_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """奇偶特征：奇数数量、偶数数量、奇偶比

        Args:
            window_data: 窗口数据DataFrame

        Returns:
            包含奇偶特征的DataFrame
        """
        features: Dict[str, np.ndarray] = {}
        row_values = window_data.values

        features["Odd_Count"] = (row_values % 2 == 1).sum(axis=1)
        features["Even_Count"] = (row_values % 2 == 0).sum(axis=1)
        features["Odd_Even_Ratio"] = features["Odd_Count"] / (features["Even_Count"] + 1e-6)

        return pd.DataFrame(features)

    def calc_size_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """大小特征：大号数量(>16)、小号数量(<=16)、大小比

        Args:
            window_data: 窗口数据DataFrame

        Returns:
            包含大小特征的DataFrame
        """
        features: Dict[str, np.ndarray] = {}
        row_values = window_data.values

        features["Big_Count"] = (row_values > 16).sum(axis=1)
        features["Small_Count"] = (row_values <= 16).sum(axis=1)
        features["Big_Small_Ratio"] = features["Big_Count"] / (features["Small_Count"] + 1e-6)

        return pd.DataFrame(features)

    def calc_consecutive_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """连号特征：连号对数、最长连号长度、连号比例

        Args:
            window_data: 窗口数据DataFrame

        Returns:
            包含连号特征的DataFrame
        """
        features: Dict[str, List] = {"Consecutive_Pairs": [], "Max_Consecutive": []}
        row_values = window_data.values

        for row in row_values:
            sorted_row = np.sort(row)
            diffs = np.diff(sorted_row)
            consecutive_pairs = int(np.sum(diffs == 1))

            max_consecutive = 1
            current = 1
            for d in diffs:
                if d == 1:
                    current += 1
                    max_consecutive = max(max_consecutive, current)
                else:
                    current = 1

            features["Consecutive_Pairs"].append(consecutive_pairs)
            features["Max_Consecutive"].append(max_consecutive)

        n_cols = len(window_data.columns)
        features["Consecutive_Ratio"] = np.array(features["Consecutive_Pairs"]) / max(n_cols - 1, 1)

        return pd.DataFrame(features)

    def calc_prime_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """质数特征：质数数量、质数比例

        Args:
            window_data: 窗口数据DataFrame

        Returns:
            包含质数特征的DataFrame
        """
        features: Dict[str, np.ndarray] = {}
        row_values = window_data.values

        features["Prime_Count"] = np.isin(row_values, list(self.primes)).sum(axis=1)
        features["Prime_Ratio"] = features["Prime_Count"] / len(window_data.columns)

        return pd.DataFrame(features)

    def calc_position_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """位置特征：最大值位置、最小值位置、中位数位置

        Args:
            window_data: 窗口数据DataFrame

        Returns:
            包含位置特征的DataFrame
        """
        features: Dict[str, np.ndarray] = {}
        row_values = window_data.values

        features["Max_Position"] = np.argmax(row_values, axis=1)
        features["Min_Position"] = np.argmin(row_values, axis=1)

        median_positions: List[int] = []
        for row in row_values:
            sorted_indices = np.argsort(row)
            median_positions.append(int(sorted_indices[len(row) // 2]))
        features["Median_Position"] = np.array(median_positions)

        return pd.DataFrame(features)

    def calc_recent_frequency(
        self, df: pd.DataFrame, window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """近期频率特征：滑动窗口内每个号码的出现频率

        Args:
            df: 原始数据DataFrame (包含红球列)
            window_size: 滑动窗口大小，默认使用配置中的 recent_freq_window

        Returns:
            添加了 Recent_Freq_{num} 列的DataFrame
        """
        window_size = window_size or FEATURE_CONFIG["recent_freq_window"]
        result = df.copy()
        red_data = result[self.red_cols].values

        for num in range(1, self.red_numbers + 1):
            mask = (red_data == num).astype(np.float32).sum(axis=1)
            result[f"Recent_Freq_{num}"] = (
                pd.Series(mask).rolling(window=window_size, min_periods=1).sum()
            )

        return result

    def calc_last_appearance(self, df: pd.DataFrame) -> pd.DataFrame:
        """遗漏特征：每个号码距离上次出现的期数

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Last_Appear_{num} 列的DataFrame
        """
        result = df.copy()
        red_data = result[self.red_cols].values
        n = len(result)

        for num in range(1, self.red_numbers + 1):
            mask = (red_data == num).astype(np.float32).sum(axis=1)
            positions = np.where(mask == 1)[0]

            if len(positions) == 0:
                result[f"Last_Appear_{num}"] = np.arange(1, n + 1, dtype=np.float32)
            else:
                expanded = np.zeros(n, dtype=np.float32)
                expanded[positions] = positions.astype(np.float32) + 1
                last_pos = np.maximum.accumulate(expanded)
                last_pos[last_pos == 0] = np.nan
                result[f"Last_Appear_{num}"] = np.nan_to_num(
                    np.arange(1, n + 1, dtype=np.float32) - last_pos,
                    nan=np.arange(1, n + 1, dtype=np.float32),
                )

        return result

    def calc_poisson_features(
        self, df: pd.DataFrame, window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """泊松分布特征：每个号码在滑动窗口内的泊松分布概率

        Args:
            df: 原始数据DataFrame (包含红球列)
            window_size: 滑动窗口大小，默认使用配置中的 poisson_window

        Returns:
            添加了 Poisson_{num} 列的DataFrame
        """
        window_size = window_size or FEATURE_CONFIG["poisson_window"]
        result = df.copy()
        red_data = result[self.red_cols].values

        for num in range(1, self.red_numbers + 1):
            mask = (red_data == num).astype(np.float32).sum(axis=1)
            freq_col = pd.Series(mask).rolling(window=window_size, min_periods=1).sum()
            lam = freq_col / window_size
            result[f"Poisson_{num}"] = poisson.pmf(1, lam).astype(np.float32)

        return result

    def calc_normal_features(
        self, df: pd.DataFrame, window_size: int = 50
    ) -> pd.DataFrame:
        """正态分布特征：红球和值的移动统计量

        Args:
            df: 原始数据DataFrame (包含红球列)
            window_size: 滚动窗口大小，默认50

        Returns:
            添加了 Sum_Mean, Sum_Std, Sum_Skew, Sum_Kurt 列的DataFrame
        """
        result = df.copy()
        result["Sum"] = result[self.red_cols].sum(axis=1)
        result["Sum_Mean"] = (
            result["Sum"].rolling(window=window_size, min_periods=10).mean().fillna(result["Sum"].mean())
        )
        result["Sum_Std"] = (
            result["Sum"].rolling(window=window_size, min_periods=10).std().fillna(result["Sum"].std())
        )
        result["Sum_Skew"] = (
            result["Sum"].rolling(window=window_size, min_periods=10).skew().fillna(0)
        )
        result["Sum_Kurt"] = (
            result["Sum"].rolling(window=window_size, min_periods=10).kurt().fillna(0)
        )

        return result

    def calc_entropy_features(
        self, df: pd.DataFrame, window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """信息熵特征：近期号码分布的信息熵

        Args:
            df: 原始数据DataFrame (包含红球列)
            window_size: 滑动窗口大小，默认使用配置中的 entropy_window

        Returns:
            添加了 Entropy 列的DataFrame
        """
        window_size = window_size or FEATURE_CONFIG["entropy_window"]
        result = df.copy()
        red_data = result[self.red_cols].values
        n = len(result)
        entropy = np.zeros(n, dtype=np.float32)

        for i in range(window_size, n):
            window = red_data[i - window_size:i].flatten()
            _, counts = np.unique(window, return_counts=True)
            probs = counts / len(window)
            entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))

        result["Entropy"] = entropy
        return result

    def calc_markov_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """马尔可夫链特征：奇偶比的转移概率

        马氏性闸门(Rocky 指示 2026-08-13, 源自阿里云 264900 方法论):
        先对奇偶比序列做卡方马氏性检验, 若不通过(α=0.05)则标记 markov_valid=False,
        下游应据此降级/弃用该特征, 防止对强随机序列硬套马尔可夫产生虚假信心。

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Markov_Prob_{state} 列 + markov_valid/markov_chi2_p 列的DataFrame
        """
        result = df.copy()
        result["Odd_Count"] = (result[self.red_cols] % 2 == 1).sum(axis=1).astype(np.int32)

        states = 7
        odd_vals = result["Odd_Count"].values

        # 模块级工具: 卡方马氏性检验(返回 p 值与是否通过)
        chi2_stat, chi2_p, is_markov = markov_chi_square_test(odd_vals, states)

        transition_matrix = np.zeros((states, states), dtype=np.float32)
        for i in range(1, len(result)):
            transition_matrix[odd_vals[i - 1], odd_vals[i]] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            transition_prob = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
            transition_prob = np.nan_to_num(transition_prob)

        for state in range(states):
            result[f"Markov_Prob_{state}"] = transition_prob[odd_vals, state]

        # 闸门信号: 下游(模型训练/特征选择)按此列决定是否采纳马尔可夫特征
        result["markov_valid"] = bool(is_markov)
        result["markov_chi2_p"] = float(chi2_p)
        return result

    def calc_hot_cold_features(
        self, df: pd.DataFrame, recent_window: Optional[int] = None
    ) -> pd.DataFrame:
        """冷热号特征：近期热门号码数、冷门号码数、冷热比

        Args:
            df: 原始数据DataFrame (包含红球列)
            recent_window: 近期窗口大小，默认使用配置中的 recent_freq_window

        Returns:
            添加了 Hot_Count, Cold_Count, Hot_Cold_Ratio 等列的DataFrame
        """
        recent_window = recent_window or FEATURE_CONFIG["recent_freq_window"]
        result = df.copy()
        red_data = result[self.red_cols].values
        n = len(result)

        freq_matrix = np.zeros((n, self.red_numbers), dtype=np.float32)
        for num in range(1, self.red_numbers + 1):
            freq_matrix[:, num - 1] = (red_data == num).astype(np.float32).sum(axis=1)

        recent_freq = np.zeros_like(freq_matrix)
        for i in range(n):
            window_start = max(0, i - recent_window + 1)
            recent_freq[i] = freq_matrix[window_start:i + 1].sum(axis=0)

        hot_threshold = np.percentile(recent_freq, 70, axis=1)
        cold_threshold = np.percentile(recent_freq, 30, axis=1)

        hot_count = np.zeros(n, dtype=np.float32)
        cold_count = np.zeros(n, dtype=np.float32)

        for i in range(n):
            nums = red_data[i]
            for num in nums:
                idx = int(num) - 1
                if recent_freq[i, idx] >= hot_threshold[i]:
                    hot_count[i] += 1
                elif recent_freq[i, idx] <= cold_threshold[i]:
                    cold_count[i] += 1

        result["Hot_Count"] = hot_count
        result["Cold_Count"] = cold_count
        result["Hot_Cold_Ratio"] = np.where(
            cold_count == 0, hot_count, hot_count / (cold_count + 1e-6)
        )

        window_size = FEATURE_CONFIG.get("position_window", 30)
        result["Hot_Count_Mean"] = (
            result["Hot_Count"].rolling(window=window_size, min_periods=5).mean().fillna(result["Hot_Count"].mean())
        )
        result["Cold_Count_Mean"] = (
            result["Cold_Count"].rolling(window=window_size, min_periods=5).mean().fillna(result["Cold_Count"].mean())
        )

        return result

    def calc_sin_cos_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """正弦余弦编码特征：多周期配对编码

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Sin_{i}_P{period}, Cos_{i}_P{period} 列的DataFrame
        """
        result = df.copy()
        periods = [self.red_numbers, 11, 3]

        for period in periods:
            for i, col in enumerate(self.red_cols):
                result[f"Sin_{i+1}_P{period}"] = np.sin(2 * np.pi * result[col] / period).astype(np.float32)
                result[f"Cos_{i+1}_P{period}"] = np.cos(2 * np.pi * result[col] / period).astype(np.float32)

        return result

    def calc_law_of_large_numbers_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """大数定律特征：每个号码的累计出现次数与理论期望的偏差

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 LLN_Deviation_{num} 等列的DataFrame
        """
        result = df.copy()
        red_data = result[self.red_cols].values
        n = len(result)

        for num in range(1, self.red_numbers + 1):
            mask = (red_data == num).astype(np.float32).sum(axis=1)
            cumulative_count = np.cumsum(mask)
            cumulative_expected = np.arange(1, n + 1, dtype=np.float32) / self.red_numbers
            result[f"LLN_Deviation_{num}"] = cumulative_count - cumulative_expected

        result["LLN_Max_Deviation"] = result[[f"LLN_Deviation_{num}" for num in range(1, self.red_numbers + 1)]].max(axis=1)
        result["LLN_Min_Deviation"] = result[[f"LLN_Deviation_{num}" for num in range(1, self.red_numbers + 1)]].min(axis=1)
        result["LLN_Abs_Deviation_Mean"] = result[[f"LLN_Deviation_{num}" for num in range(1, self.red_numbers + 1)]].abs().mean(axis=1)

        return result

    def calc_sum_interval_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """和值区间特征：和值所在区间及各区间频率

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Sum_Int_{i}, Sum_Int_{i}_Freq 列的DataFrame
        """
        result = df.copy()
        if "Sum" not in result.columns:
            result["Sum"] = result[self.red_cols].sum(axis=1)

        intervals = [(30, 50), (51, 70), (71, 90), (91, 110), (111, 130), (131, 150)]
        interval_labels = [f"Sum_Int_{i}" for i in range(1, len(intervals) + 1)]

        for label, (low, high) in zip(interval_labels, intervals):
            result[label] = ((result["Sum"] >= low) & (result["Sum"] <= high)).astype(np.float32)

        for label in interval_labels:
            result[f"{label}_Freq"] = (
                result[label].rolling(window=50, min_periods=5).mean().fillna(result[label].mean())
            )

        return result

    def calc_interval_distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """区间分布特征：各区间(1-11, 12-22, 23-33)号码数量及统计量

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Int_Dist_{i}, Int_Dist_Max, Int_Dist_Min, Int_Dist_Std 等列的DataFrame
        """
        result = df.copy()
        red_data = result[self.red_cols].values

        intervals = [(1, 11), (12, 22), (23, 33)]
        interval_labels = ["Int_Dist_1", "Int_Dist_2", "Int_Dist_3"]

        for label, (low, high) in zip(interval_labels, intervals):
            result[label] = np.sum((red_data >= low) & (red_data <= high), axis=1).astype(np.float32)

        result["Int_Dist_Max"] = result[interval_labels].max(axis=1)
        result["Int_Dist_Min"] = result[interval_labels].min(axis=1)
        result["Int_Dist_Std"] = result[interval_labels].std(axis=1)

        window_size = FEATURE_CONFIG.get("position_window", 30)
        for label in interval_labels:
            result[f"{label}_Mean"] = (
                result[label].rolling(window=window_size, min_periods=5).mean().fillna(result[label].mean())
            )

        return result

    def calc_position_stats(
        self, df: pd.DataFrame, window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """位置统计特征：每个位置号码的滚动统计量

        Args:
            df: 原始数据DataFrame (包含红球列)
            window_size: 滚动窗口大小，默认使用配置中的 position_window

        Returns:
            添加了 Pos_{i}_Mean, Pos_{i}_Std, Pos_{i}_Recent 列的DataFrame
        """
        window_size = window_size or FEATURE_CONFIG["position_window"]
        result = df.copy()

        for i, col in enumerate(self.red_cols):
            result[f"Pos_{i+1}_Mean"] = (
                result[col].rolling(window=window_size, min_periods=5).mean().fillna(result[col].mean())
            )
            result[f"Pos_{i+1}_Std"] = (
                result[col].rolling(window=window_size, min_periods=5).std().fillna(result[col].std())
            )
            result[f"Pos_{i+1}_Recent"] = (
                result[col].rolling(window=3, min_periods=1).mean().fillna(result[col].mean())
            )

        return result

    def calc_odd_even_per_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """每个位置的奇偶标志特征

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Odd_Even_{i} 列的DataFrame
        """
        result = df.copy()
        for i, col in enumerate(self.red_cols):
            result[f"Odd_Even_{i+1}"] = (result[col] % 2).astype(np.float32)
        return result

    def calc_size_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """大小分布特征：大号数量、小号数量、大小比（带滚动统计）

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Big_Count, Small_Count, Big_Small_Ratio 等列的DataFrame
        """
        result = df.copy()
        red_data = result[self.red_cols].values

        result["Big_Count"] = np.sum(red_data > 16, axis=1).astype(np.float32)
        result["Small_Count"] = np.sum(red_data <= 16, axis=1).astype(np.float32)
        result["Big_Small_Ratio"] = np.where(
            result["Small_Count"] == 0, result["Big_Count"], result["Big_Count"] / result["Small_Count"]
        )

        window_size = FEATURE_CONFIG.get("position_window", 30)
        result["Big_Count_Mean"] = (
            result["Big_Count"].rolling(window=window_size, min_periods=5).mean().fillna(result["Big_Count"].mean())
        )
        result["Small_Count_Mean"] = (
            result["Small_Count"].rolling(window=window_size, min_periods=5).mean().fillna(result["Small_Count"].mean())
        )

        return result

    def calc_prime_with_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """质数特征（带滚动统计）：质数数量、质数比例及其移动平均

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Prime_Count, Prime_Ratio, Prime_Count_Mean 等列的DataFrame
        """
        result = df.copy()
        red_data = result[self.red_cols].values
        prime_mask = np.isin(red_data, list(self.primes)).astype(np.float32)

        result["Prime_Count"] = prime_mask.sum(axis=1)
        result["Prime_Ratio"] = result["Prime_Count"] / len(self.red_cols)

        window_size = FEATURE_CONFIG.get("position_window", 30)
        result["Prime_Count_Mean"] = (
            result["Prime_Count"].rolling(window=window_size, min_periods=5).mean().fillna(result["Prime_Count"].mean())
        )
        result["Prime_Ratio_Mean"] = (
            result["Prime_Ratio"].rolling(window=window_size, min_periods=5).mean().fillna(result["Prime_Ratio"].mean())
        )

        return result

    def calc_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础统计特征：和值、奇偶比、大小比

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Sum, OddRatio, BigRatio 列的DataFrame
        """
        result = df.copy()
        all_reds = result[self.red_cols]
        result["Sum"] = all_reds.sum(axis=1)
        result["OddRatio"] = (all_reds % 2 == 1).sum(axis=1) / len(self.red_cols)
        result["BigRatio"] = (all_reds >= 17).sum(axis=1) / len(self.red_cols)
        return result

    def calc_onehot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """上期红球号码的One-Hot编码特征

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 Last_Draw_{num} 列的DataFrame
        """
        result = df.copy()
        all_reds = result[self.red_cols].shift(1).values

        for i in range(1, self.red_numbers + 1):
            result[f"Last_Draw_{i}"] = (all_reds == i).any(axis=1).astype(np.float32)

        return result

    def calc_statistical_features_extended(
        self, df: pd.DataFrame, window_size: int = 100
    ) -> pd.DataFrame:
        """Hot/Cold/Omission统计特征（扩展版）

        Args:
            df: 原始数据DataFrame (包含红球列)
            window_size: 滑动窗口大小

        Returns:
            添加了 Hot_Count, Cold_Count, Max_Omission, Avg_Omission 列的DataFrame
        """
        result = df.copy()
        red_data = result[self.red_cols].values
        n = len(result)

        max_omissions = np.zeros(n, dtype=np.float32)
        avg_omissions = np.zeros(n, dtype=np.float32)

        for num in range(1, self.red_numbers + 1):
            positions = np.where(red_data == num)[0]
            last_seen = np.zeros(n, dtype=np.int32)

            ptr = 0
            for i in range(n):
                if ptr < len(positions) and positions[ptr] == i:
                    last_seen[i] = i
                    ptr += 1
                elif ptr > 0:
                    last_seen[i] = last_seen[i - 1]
                else:
                    last_seen[i] = -1

            omissions = np.arange(n) - last_seen
            mask = last_seen == -1
            omissions[mask] = np.where(mask)[0] + 1

            max_omissions = np.maximum(max_omissions, omissions)
            avg_omissions += omissions

        avg_omissions /= self.red_numbers

        hot_counts = np.zeros(n, dtype=np.float32)
        cold_counts = np.zeros(n, dtype=np.float32)

        for i in range(n):
            start = max(0, i - 6)
            if i >= 6:
                recent_data = red_data[start:i].flatten()
                unique, counts = np.unique(recent_data, return_counts=True)
                hot_counts[i] = np.sum(counts >= 2)
                cold_counts[i] = self.red_numbers - len(unique)

        result["Hot_Count"] = hot_counts
        result["Cold_Count"] = cold_counts
        result["Max_Omission"] = max_omissions
        result["Avg_Omission"] = avg_omissions

        return result

    def calc_markov_extended(self, df: pd.DataFrame) -> pd.DataFrame:
        """马尔可夫链扩展特征：奇偶比和大小比的转移概率

        Args:
            df: 包含 OddRatio, BigRatio 列的DataFrame

        Returns:
            添加了 Markov_Odd_Prob_{i}, Markov_Big_Prob_{i} 列的DataFrame
        """
        result = df.copy()
        states = FEATURE_CONFIG.get("markov_states", 7)

        required_cols = ["OddRatio", "BigRatio"]
        for col in required_cols:
            if col not in result.columns:
                result[col] = 0.0

        red_count = len(self.red_cols)

        def _safe_state(val: float) -> int:
            """安全地将比率值转为状态索引，防止越界"""
            if pd.isna(val):
                return 0
            state = int(val * red_count)
            return max(0, min(state, states - 1))

        markov_matrix_odd = np.zeros((states, states), dtype=np.float32)
        markov_matrix_big = np.zeros((states, states), dtype=np.float32)

        for i in range(1, len(result)):
            prev_odd_state = _safe_state(result.loc[i - 1, "OddRatio"])
            curr_odd_state = _safe_state(result.loc[i, "OddRatio"])
            prev_big_state = _safe_state(result.loc[i - 1, "BigRatio"])
            curr_big_state = _safe_state(result.loc[i, "BigRatio"])

            markov_matrix_odd[prev_odd_state, curr_odd_state] += 1
            markov_matrix_big[prev_big_state, curr_big_state] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            markov_prob_odd = markov_matrix_odd / markov_matrix_odd.sum(axis=1, keepdims=True)
            markov_prob_big = markov_matrix_big / markov_matrix_big.sum(axis=1, keepdims=True)
            markov_prob_odd = np.nan_to_num(markov_prob_odd)
            markov_prob_big = np.nan_to_num(markov_prob_big)

        next_odd_probs: List[np.ndarray] = []
        next_big_probs: List[np.ndarray] = []
        for i in range(len(result)):
            curr_odd = _safe_state(result.loc[i, "OddRatio"])
            curr_big = _safe_state(result.loc[i, "BigRatio"])
            next_odd_probs.append(markov_prob_odd[curr_odd])
            next_big_probs.append(markov_prob_big[curr_big])

        odd_df = pd.DataFrame(
            next_odd_probs, columns=[f"Markov_Odd_Prob_{i}" for i in range(states)]
        )
        big_df = pd.DataFrame(
            next_big_probs, columns=[f"Markov_Big_Prob_{i}" for i in range(states)]
        )

        result = pd.concat([result, odd_df, big_df], axis=1)
        return result

    def generate_regression_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成下一期的回归目标特征

        Args:
            df: 包含统计特征列的DataFrame

        Returns:
            添加了 Next_{feature} 回归目标列的DataFrame
        """
        result = df.copy()
        regression_targets = [
            "Sum", "OddRatio", "BigRatio",
            "Hot_Count", "Cold_Count", "Max_Omission", "Avg_Omission",
        ]

        for col in regression_targets:
            if col in result.columns:
                result[f"Next_{col}"] = result[col].shift(-1)

        return result

    def compute_all_features(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        window_size: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """运行所有特征计算的主入口方法

        按顺序调用所有特征计算方法，整合为完整的特征集。

        Args:
            df: 原始数据DataFrame
            target_cols: 目标列名列表，为None时使用配置
            window_size: 滑动窗口大小，为None时使用配置
            **kwargs: 传递给各特征方法的额外参数

        Returns:
            包含所有计算特征的DataFrame，已去除NaN行
        """
        import time

        start_time = time.time()
        result = df.copy()

        target_cols = target_cols or self.target_cols
        window_size = window_size or FEATURE_CONFIG["window_size"]

        feature_funcs = [
            ("基础统计", self.calc_base_features),
            ("位置奇偶", self.calc_odd_even_per_position),
            ("近期频率", self.calc_recent_frequency),
            ("遗漏特征", self.calc_last_appearance),
            ("泊松分布", self.calc_poisson_features),
            ("正态分布", self.calc_normal_features),
            ("信息熵", self.calc_entropy_features),
            ("马尔可夫链", self.calc_markov_features),
            ("位置统计", self.calc_position_stats),
            ("和值区间", self.calc_sum_interval_features),
            ("冷热号", self.calc_hot_cold_features),
            ("区间分布", self.calc_interval_distribution_features),
            ("质数特征", self.calc_prime_with_stats),
            ("大小分布", self.calc_size_distribution),
            ("正弦余弦", self.calc_sin_cos_features),
            ("大数定律", self.calc_law_of_large_numbers_features),
            ("扩展统计", lambda df=result: self.calc_statistical_features_extended(df, window_size)),
            ("One-Hot", self.calc_onehot_features),
            ("马尔可夫扩展", self.calc_markov_extended),
        ]

        for name, func in feature_funcs:
            t_start = time.time()
            try:
                result = func(result)
                elapsed = time.time() - t_start
                print(f"  [耗时 {elapsed:.2f}s] {name}特征")
            except Exception as e:
                print(f"  [失败] {name}特征计算失败: {e}")

        result = self.generate_regression_targets(result)

        total_time = time.time() - start_time
        print(f"\n特征工程总耗时: {total_time:.2f}秒")
        print(f"有效记录数: {len(result)}")

        return result.dropna().reset_index(drop=True)

    def calc_ac_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """AC 值特征：每期 6 个红球两两差绝对值去重数 - 5。

        AC 值衡量号码分散度，范围 0~10，越分散越大：
          [1,2,3,4,5,6]    -> 两两差 {1,2,3,4,5} 去重 5 个 - 5 = 0
          [1,7,13,19,25,31] -> 两两差 15 个全部不同 = 15 - 5 = 10

        Args:
            df: 原始数据DataFrame (包含红球列)

        Returns:
            添加了 AC_Value 列(float)的DataFrame
        """
        result = df.copy()
        red_data = result[self.red_cols].values.astype(int)
        ac = np.zeros(len(result), dtype=np.float64)
        for i, row in enumerate(red_data):
            diffs = {abs(int(a) - int(b)) for j, a in enumerate(row) for b in row[j + 1:]}
            ac[i] = len(diffs) - 5
        result["AC_Value"] = ac
        return result

    # 严格去冗余/去泄漏的统一特征白名单（compact 模式）。
    # 注意：Next_*（shift(-1) 未来标签）、Last_Draw_*（shift(1) 上期 one-hot）、
    # LLN_Deviation_*（与 Last_Appear/LLN_Abs 共线）等均剔除，避免信息泄漏与共线。
    UNIFIED_KEEP = [
        "Sum", "OddRatio", "BigRatio", "Odd_Count",
        *[f"Recent_Freq_{i}" for i in range(1, 34)],
        *[f"Last_Appear_{i}" for i in range(1, 34)],
        *[f"Poisson_{i}" for i in range(1, 34)],
        "Sum_Mean", "Sum_Std", "Sum_Skew", "Sum_Kurt", "Entropy",
        "Int_Dist_1", "Int_Dist_2", "Int_Dist_3",
        "Int_Dist_Max", "Int_Dist_Min", "Int_Dist_Std",
        *[f"Sum_Int_{i}" for i in range(1, 7)],
        "Hot_Count", "Cold_Count", "Hot_Cold_Ratio",
        "Hot_Count_Mean", "Cold_Count_Mean",
        "Max_Omission", "Avg_Omission",
        "Prime_Count", "Prime_Ratio", "Prime_Count_Mean",
        "Big_Count", "Big_Small_Ratio", "Big_Count_Mean",
        "Consecutive_Pairs", "Max_Consecutive",
        *[f"Markov_Odd_Prob_{i}" for i in range(7)],
        *[f"Markov_Big_Prob_{i}" for i in range(7)],
        *[f"Pos_{i}_Mean" for i in range(1, 7)],
        *[f"Pos_{i}_Std" for i in range(1, 7)],
        "LLN_Abs_Deviation_Mean",
    ]

    def build_unified_features(self, df: pd.DataFrame, mode: str = "compact",
                               keep_override: Optional[List[str]] = None) -> pd.DataFrame:
        """统一特征入口：所有模型共用，避免 RF/LGBM 与 LSTM/CNN 特征空间不一致。

        Args:
            df: 原始数据(含 Red1..Red6, Blue1 等列)
            mode: "full" 返回 compute_all_features 全部列(含冗余,仅供调试)；
                  "compact" 返回去冗余/去泄漏白名单列(默认,生产用)。
            keep_override: 可选追加白名单列列表(在 UNIFIED_KEEP 基础上追加)。
                  feature 开关(如 AC_Value)开启时由调用方传入对应列名；
                  为 None 时行为与旧版完全一致(向后兼容)。

        Returns:
            清洗后的特征 DataFrame（drop 掉含 NaN 的行），列顺序固定。
        """
        full = self.compute_all_features(df)
        if mode == "full":
            return full.dropna().reset_index(drop=True)
        keep = [c for c in self.UNIFIED_KEEP if c in full.columns]
        if keep_override:
            missing = [c for c in keep_override if c not in full.columns]
            # AC_Value 由 calc_ac_features 按需生成(默认管线不含该列, 由特征开关驱动)
            if "AC_Value" in missing:
                full = self.calc_ac_features(full)
            keep += [c for c in keep_override if c not in keep and c in full.columns]
        miss = [c for c in self.UNIFIED_KEEP if c not in full.columns]
        if miss:
            print(f"  [警告] 白名单缺失列(已跳过): {miss}")
        out = full[keep].dropna().reset_index(drop=True)
        return out