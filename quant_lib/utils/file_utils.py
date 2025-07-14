"""
文件工具模块

提供文件操作相关的工具函数。
"""

import os
import pandas as pd
import pickle
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

# 获取模块级别的logger
logger = logging.getLogger(__name__)


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {path}")
    return path


def save_to_csv(df: pd.DataFrame, 
               filepath: Union[str, Path], 
               index: bool = True, 
               encoding: str = 'utf-8') -> None:
    """
    保存DataFrame到CSV文件
    
    Args:
        df: 要保存的DataFrame
        filepath: 文件路径
        index: 是否保存索引
        encoding: 文件编码
    """
    filepath = Path(filepath)
    ensure_dir_exists(filepath.parent)
    
    try:
        df.to_csv(filepath, index=index, encoding=encoding)
        logger.info(f"数据已保存至: {filepath}")
    except Exception as e:
        logger.error(f"保存CSV文件失败: {e}")


def save_to_excel(df: pd.DataFrame, 
                 filepath: Union[str, Path], 
                 sheet_name: str = 'Sheet1',
                 index: bool = True) -> None:
    """
    保存DataFrame到Excel文件
    
    Args:
        df: 要保存的DataFrame
        filepath: 文件路径
        sheet_name: 工作表名称
        index: 是否保存索引
    """
    filepath = Path(filepath)
    ensure_dir_exists(filepath.parent)
    
    try:
        df.to_excel(filepath, sheet_name=sheet_name, index=index)
        logger.info(f"数据已保存至: {filepath}")
    except Exception as e:
        logger.error(f"保存Excel文件失败: {e}")


def save_to_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    保存对象到Pickle文件
    
    Args:
        obj: 要保存的对象
        filepath: 文件路径
    """
    filepath = Path(filepath)
    ensure_dir_exists(filepath.parent)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"对象已保存至: {filepath}")
    except Exception as e:
        logger.error(f"保存Pickle文件失败: {e}")


def load_from_pickle(filepath: Union[str, Path]) -> Any:
    """
    从Pickle文件加载对象
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的对象
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"文件不存在: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"对象已从 {filepath} 加载")
        return obj
    except Exception as e:
        logger.error(f"加载Pickle文件失败: {e}")
        return None


def save_to_json(obj: Dict, 
                filepath: Union[str, Path], 
                indent: int = 4,
                ensure_ascii: bool = False) -> None:
    """
    保存对象到JSON文件
    
    Args:
        obj: 要保存的对象
        filepath: 文件路径
        indent: 缩进空格数
        ensure_ascii: 是否确保ASCII编码
    """
    filepath = Path(filepath)
    ensure_dir_exists(filepath.parent)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)
        logger.info(f"对象已保存至: {filepath}")
    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")


def load_from_json(filepath: Union[str, Path]) -> Dict:
    """
    从JSON文件加载对象
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的对象
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"文件不存在: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        logger.info(f"对象已从 {filepath} 加载")
        return obj
    except Exception as e:
        logger.error(f"加载JSON文件失败: {e}")
        return {}


def save_to_yaml(obj: Dict, 
                filepath: Union[str, Path], 
                default_flow_style: bool = False) -> None:
    """
    保存对象到YAML文件
    
    Args:
        obj: 要保存的对象
        filepath: 文件路径
        default_flow_style: 默认流样式
    """
    filepath = Path(filepath)
    ensure_dir_exists(filepath.parent)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(obj, f, default_flow_style=default_flow_style, allow_unicode=True)
        logger.info(f"对象已保存至: {filepath}")
    except Exception as e:
        logger.error(f"保存YAML文件失败: {e}")


def load_from_yaml(filepath: Union[str, Path]) -> Dict:
    """
    从YAML文件加载对象
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的对象
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"文件不存在: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            obj = yaml.safe_load(f)
        logger.info(f"对象已从 {filepath} 加载")
        return obj
    except Exception as e:
        logger.error(f"加载YAML文件失败: {e}")
        return {} 