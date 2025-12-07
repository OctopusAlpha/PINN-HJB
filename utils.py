import yaml
import torch
import logging
import os
from logging.handlers import RotatingFileHandler

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg['device'] == 'auto':
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg

def get_logger(name: str, log_file: str = None, level=logging.INFO, max_bytes=10*1024*1024, backup_count=5):
    """
    获取一个配置好的 logger 实例（带控制台和文件输出）

    Args:
        name (str): logger 名称（通常用 __name__）
        log_file (str, optional): 日志文件路径，如 'logs/app.log'。若为 None，则只输出到控制台。
        level: 日志级别，默认 INFO
        max_bytes: 单个日志文件最大大小（默认 10MB）
        backup_count: 保留的旧日志文件数量（默认 5 个）

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler（重要！）
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # 格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（可选）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger