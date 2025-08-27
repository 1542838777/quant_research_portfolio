import numpy as np
from dataclasses import dataclass


# --- 模拟Config类 ---
@dataclass
class MockConfig:
    enable_turnover_penalty: bool = True
    turnover_weight: float = 0.25
    reward_turnover_rate_daily: float = 0.0025
    max_turnover_rate_daily: float = 0.007
    penalty_slope_daily: float = 45.0
    heavy_penalty_slope_daily: float = 100.0
    base_turnover_multiplier_floor: float = 0.1
    turnover_vol_threshold_ratio: float = 0.5
    turnover_vol_penalty_factor: float = 0.2
    turnover_trend_sensitivity: float = 50.0
    final_multiplier_min: float = 0.1
    final_multiplier_max: float = 1.2


class FactorScorer:
    def __init__(self):
        self.config = MockConfig()

    def _calculate_turnover_adjusted_score_v3(self, base_score: float, turnover_stats: dict) -> float:
        """
        计算基于多维度换手率指标的调整后评分 (V3 - 最终生产版)
        """
        if not self.config.enable_turnover_penalty:
            return base_score

        epsilon = 1e-8
        avg_daily_rank_change = turnover_stats.get('avg_daily_rank_change', 0.01)

        reward_rate_daily = self.config.reward_turnover_rate_daily
        max_rate_daily = self.config.max_turnover_rate_daily
        penalty_slope = self.config.penalty_slope_daily
        heavy_penalty_slope = self.config.heavy_penalty_slope_daily

        if avg_daily_rank_change <= reward_rate_daily:
            base_turnover_multiplier = 1.0 + (avg_daily_rank_change / (reward_rate_daily + epsilon)) * 0.1
        elif avg_daily_rank_change <= max_rate_daily:
            base_turnover_multiplier = 1.1 - (avg_daily_rank_change - reward_rate_daily) * penalty_slope
        else:
            boundary_multiplier = 1.1 - (max_rate_daily - reward_rate_daily) * penalty_slope
            excess_turnover = avg_daily_rank_change - max_rate_daily
            base_turnover_multiplier = boundary_multiplier - excess_turnover * heavy_penalty_slope

        base_turnover_multiplier = max(base_turnover_multiplier, self.config.base_turnover_multiplier_floor)

        volatility = turnover_stats.get('daily_turnover_volatility', 0)
        volatility_threshold_ratio = self.config.turnover_vol_threshold_ratio
        volatility_penalty_factor = self.config.turnover_vol_penalty_factor

        volatility_penalty_multiplier = 1.0

        ratio = volatility / (avg_daily_rank_change + epsilon)
        if ratio > volatility_threshold_ratio:
            excess_ratio = ratio - volatility_threshold_ratio
            penalty = excess_ratio * volatility_penalty_factor
            volatility_penalty_multiplier = max(0.8, 1.0 - penalty)

        trend = turnover_stats.get('daily_turnover_trend', 0)
        trend_penalty_multiplier = 1.0

        if trend > 0:
            relative_trend = trend / (avg_daily_rank_change + epsilon)
            sensitivity = self.config.turnover_trend_sensitivity

            trend_penalty_multiplier = np.exp(-relative_trend * sensitivity)
            trend_penalty_multiplier = max(0.7, trend_penalty_multiplier)

        total_turnover_multiplier = base_turnover_multiplier * volatility_penalty_multiplier * trend_penalty_multiplier

        weight = self.config.turnover_weight
        final_multiplier = (1 - weight) + weight * total_turnover_multiplier

        final_multiplier = np.clip(
            final_multiplier,
            self.config.final_multiplier_min,
            self.config.final_multiplier_max
        )

        adjusted_score = base_score * final_multiplier
        return adjusted_score
turnover_stats_1 = {
    'avg_daily_rank_change': 0.0015,
    'daily_turnover_volatility': 0.0005,
    'daily_turnover_trend': -0.000001,
}
turnover_stats_2 = {
    'avg_daily_rank_change': 0.006,
    'daily_turnover_volatility': 0.002,
    'daily_turnover_trend': 0.0,
}
turnover_stats_3 = {
    'avg_daily_rank_change': 0.005,
    'daily_turnover_volatility': 0.002,
    'daily_turnover_trend': 0.00005, # 趋势为正，衰减信号
}

turnover_stats_4 = {
    'avg_daily_rank_change': 0.01,
    'daily_turnover_volatility': 0.008, # 波动率相对均值很高
    'daily_turnover_trend': 0.0,
}
turnover_stats_5 = {
    'avg_daily_rank_change': 0.026,
    'daily_turnover_volatility': 0.0205,
    'daily_turnover_trend': -0.0013, # 趋势显著为负
}
# --- 执行测试 ---
if __name__ == '__main__':
    scorer = FactorScorer()
    base_score = 1.0

    scenarios = {
        "1. 价值之王": turnover_stats_1,
        "2. 稳健动量": turnover_stats_2,
        "3. 昔日明星": turnover_stats_3,
        "4. 狂野牛仔": turnover_stats_4,
        "5. 你的案例": turnover_stats_5,
    }

    print(f"{'因子画像':<15} | {'调整后分数':<15}")
    print("-" * 33)

    for name, stats in scenarios.items():
        adjusted_score = scorer._calculate_turnover_adjusted_score_v3(base_score, stats)
        print(f"{name:<15} | {adjusted_score:<15.4f}")