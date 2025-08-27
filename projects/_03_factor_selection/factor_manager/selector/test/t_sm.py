import numpy as np
import logging

# --- 设置一个简单的日志记录器，以便看到警告信息 ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger()


# --- 你提供的函数 ---
def _convert_scores_to_weights(factor_scores: dict[str, float]) -> dict[str, float]:
    """将得分转换为权重"""
    if not factor_scores:
        return {}

    scores = np.array(list(factor_scores.values()))

    # 过滤掉得分过低的因子
    valid_mask = scores > 0.1
    if not valid_mask.any():
        logger.warning("⚠️ 所有因子得分都过低，使用等权重")
        return {name: 1.0 / len(factor_scores) for name in factor_scores.keys()}

    # 对有效因子应用 softmax
    valid_scores = scores[valid_mask]
    valid_names = [name for i, name in enumerate(factor_scores.keys()) if valid_mask[i]]

    # 温度参数控制权重集中度
    temperature = 2.0
    exp_scores = np.exp(valid_scores / temperature)
    softmax_weights = exp_scores / exp_scores.sum()

    # 为了便于观察，将权重四舍五入到小数点后4位
    rounded_weights = np.round(softmax_weights, 4)

    return dict(zip(valid_names, rounded_weights))


# --- 测试用例 ---
if __name__ == '__main__':
    print("--- 单元测试: _convert_scores_to_weights ---")

    # === 场景 1: 标准情况 - 所有因子得分良好 ===
    print("\n[场景 1: 标准情况]")
    scores_1 = {'Factor_Momentum': 0.8, 'Factor_Value': 0.6, 'Factor_Quality': 0.5}
    weights_1 = _convert_scores_to_weights(scores_1)
    print(f"输入得分: {scores_1}")
    print(f"输出权重: {weights_1}")
    print(f"权重总和: {sum(weights_1.values()):.4f}")
    print(">> 解读: 得分最高的动量因子获得了最高权重，但价值和质量因子也获得了显著权重。")

    # === 场景 2: 因子过滤 - 存在一个得分过低的因子 ===
    print("\n[场景 2: 因子过滤]")
    scores_2 = {'Factor_Momentum': 0.9, 'Factor_Value': 0.7, 'Factor_Reversal_Weak': 0.05}
    weights_2 = _convert_scores_to_weights(scores_2)
    print(f"输入得分: {scores_2}")
    print(f"输出权重: {weights_2}")
    print(f"权重总和: {sum(weights_2.values()):.4f}")
    print(">> 解读: 得分仅为0.05的反转因子被成功过滤掉，未参与权重计算。")

    # === 场景 3: 全部淘汰 - 所有因子得分都过低 ===
    print("\n[场景 3: 全部淘汰]")
    scores_3 = {'Factor_X': 0.08, 'Factor_Y': 0.06, 'Factor_Z': 0.02}
    weights_3 = _convert_scores_to_weights(scores_3)
    print(f"输入得分: {scores_3}")
    print(f"输出权重: {weights_3}")
    print(f"权重总和: {sum(weights_3.values()):.4f}")
    print(">> 解读: 由于所有因子得分都低于0.1的门槛，系统触发了警告并回退到等权重策略。")

    # === 场景 4: 温度效应 - 对比不同温度下的权重分配 ===
    print("\n[场景 4: 温度效应]")


    # 我们将临时修改函数内部的温度来做演示

    # 4a. 低温 (T=0.5) - 更“尖锐”
    def convert_low_temp(scores_dict):
        # (此处省略函数主体，仅修改温度)
        scores = np.array(list(scores_dict.values()))
        temperature = 0.5
        exp_scores = np.exp(scores / temperature)
        softmax_weights = exp_scores / exp_scores.sum()
        return dict(zip(scores_dict.keys(), np.round(softmax_weights, 4)))


    weights_4a = convert_low_temp(scores_1)
    print(f"输入得分: {scores_1}")
    print(f"低温 (T=0.5) 输出权重: {weights_4a}")
    print(">> 解读: 在低温下，权重分配非常集中，动量因子的权重被急剧放大。")


    # 4b. 高温 (T=5.0) - 更“平滑”
    def convert_high_temp(scores_dict):
        # (此处省略函数主体，仅修改温度)
        scores = np.array(list(scores_dict.values()))
        temperature = 5.0
        exp_scores = np.exp(scores / temperature)
        softmax_weights = exp_scores / exp_scores.sum()
        return dict(zip(scores_dict.keys(), np.round(softmax_weights, 4)))


    weights_4b = convert_high_temp(scores_1)
    print(f"高温 (T=5.0) 输出权重: {weights_4b}")
    print(">> 解读: 在高温下，各因子权重差异显著缩小，结果更接近等权重。")