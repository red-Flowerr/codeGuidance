def compute_steps(epoch_list, config='only'):
    """
    计算每个 epoch 对应的训练 step，并四舍五入到最近 2000 的倍数

    Args:
        epoch_list (list[int]): 需要计算的 epoch 列表
        config (str): 'only' 或其他配置
                      'only' step_per_epoch=7262
                      其他 config step_per_epoch=12887

    Returns:
        dict: {epoch: step_rounded}
    """
    # 设置 step per epoch
    if config == 'only':
        step_per_epoch = 7262
    else:
        step_per_epoch = 12887

    step_dict = {}
    for e in epoch_list:
        raw_step = e * step_per_epoch
        # 四舍五入到最近 2000 的倍数
        rounded_step = round(raw_step / 2000) * 2000
        step_dict[e] = rounded_step

    return step_dict

# 示例
epochs = [1, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]
print("only config:", compute_steps(epochs, config='only'))
print("else config:", compute_steps(epochs, config='other'))
# only config: {1: 8000, 2: 14000, 4: 30000, 6: 44000, 8: 58000, 10: 72000, 12: 88000, 13: 94000, 14: 102000, 15: 108000, 16: 116000, 17: 124000, 18: 130000, 19: 138000, 20: 146000}
# else config: {1: 12000, 2: 26000, 4: 52000, 6: 78000, 8: 104000, 10: 128000, 12: 154000, 13: 168000, 14: 180000, 15: 194000, 16: 206000, 17: 220000, 18: 232000, 19: 244000, 20: 258000}