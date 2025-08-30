import pandas as pd
import numpy as np
from typing import List


def convert_target_to_sequential_percents(target_weights: List[float]) -> List[float]:
    """
    将目标组合权重转换为顺序花费权重。

    Args:
        target_weights: 一个列表，包含每个资产的目标权重 (例如 [0.5, 0.5, 0.0])。
                        所有权重的总和必须小于或等于 1.0。

    Returns:
        一个列表，包含转换后的顺序花费权重。

    Raises:
        ValueError: 如果目标权重的总和大于 1.0。
    """
    if np.sum(target_weights) > 1.00001:  # 加上一个小的容差
        raise ValueError("目标权重的总和不能大于1.0")

    sequential_percents = []
    cash_ratio_remaining = 1.0

    for weight in target_weights:
        if cash_ratio_remaining > 1e-9:  # 避免除以零
            # 根据公式计算当前订单应占剩余资金的比例
            order_percent = weight / cash_ratio_remaining
            sequential_percents.append(order_percent)

            # 更新剩余资金比例
            cash_ratio_remaining -= weight
        else:
            # 如果理论上已没有现金，后续订单比例为0
            sequential_percents.append(0.0)

    return sequential_percents

def convert_to_sequential_percents(target_weights: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [convert_target_to_sequential_percents(row) for _, row in target_weights.iterrows()]
    )


def t_demo():
    # --- 演示你的例子 ---

    # 例子 1: 0.5, 0.5, 0.0
    target_1 = [0.5, 0.5, 0.0]
    sequential_1 = convert_target_to_sequential_percents(target_1)
    print(f"目标权重: {target_1}")
    print(f"转换结果: {sequential_1}  <-- 第一个0.5, 第二个是 0.5/(1-0.5)=1.0")

    print("-" * 30)

    # 例子 2: 0.5, 0.25, 0.25
    target_2 = [0.5, 0.25, 0.25]
    sequential_2 = convert_target_to_sequential_percents(target_2)
    print(f"目标权重: {target_2}")
    print(f"转换结果: {sequential_2} <-- 分别是 0.5, 0.25/(1-0.5)=0.5, 0.25/(1-0.5-0.25)=1.0")

    print("-" * 30)

    # 例子 3: 0.1, 0.4, 0.5
    target_3 = [0.1, 0.4, 0.5]
    sequential_3 = convert_target_to_sequential_percents(target_3)
    print(f"目标权重: {target_3}")
    print(f"转换结果: {sequential_3} <-- 分别是 0.1, 0.4/(1-0.1)≈0.444, 0.5/(1-0.1-0.4)=1.0")
    target_4 = [0, 0.4, 0.5, 0.1, 0, 0]
    sequential_4 = convert_target_to_sequential_percents(target_4)
    print(f"目标权重: {target_4}")
    print(f"转换结果: {sequential_4} <-- 分别是 0, 0.4, 0.83333 ,1,0,0")



if __name__ == '__main__':
    t_demo()