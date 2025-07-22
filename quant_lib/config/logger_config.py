import logging

def setup_logger(name: str = None, level: str = 'INFO') -> logging.Logger:
    """设置日志配置，避免重复配置"""
    logger_name = name or 'quant_lib'
    logger = logging.getLogger(logger_name)

    # 如果logger已经有handlers，直接返回
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(getattr(logging, level.upper()))

    # 创建handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # 添加handler
    logger.addHandler(handler)

    # 防止向上传播（避免重复输出）
    logger.propagate = False

    return logger
