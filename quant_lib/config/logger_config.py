import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

def setup_logger(
    name: str = "",
    log_dir: Path = Path("workspace/logs"),
    level: int = logging.INFO,
    when: str = "midnight",   # 每天凌晨切分
    backup_count: int = 7     # 保留最近7天日志
):
    """
    初始化并返回一个带时间戳、控制台与文件输出的 Logger 实例。

    Parameters:
    - name: logger 名称（模块名）
    - log_dir: 日志输出目录
    - level: logging 级别
    - when: 切分周期，默认按天
    - backup_count: 保留的历史文件数量

    Returns:
    - logging.Logger 对象
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name if name else 'app'}.log"

    # 日志格式
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 获取 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # 文件输出 + 按时间滚动
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when=when,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
