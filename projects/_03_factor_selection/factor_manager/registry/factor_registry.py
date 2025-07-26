"""
因子注册表模块

负责管理因子的注册、查询和元数据存储。
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from enum import Enum
from datetime import datetime

from quant_lib import logger


# 配置日志


class FactorCategory(Enum):
    """因子类别枚举"""
    VALUE = "价值"
    GROWTH = "成长"
    QUALITY = "质量"
    MOMENTUM = "动量"
    VOLATILITY = "波动率"
    LIQUIDITY = "流动性"
    SENTIMENT = "情绪"
    TECHNICAL = "技术"
    FUNDAMENTAL = "基本面"
    MACRO = "宏观"
    CUSTOM = "自定义"


class FactorMetadata:
    """因子元数据类"""
    
    def __init__(self, 
                 name: str,
                 category: Union[str, FactorCategory],
                 description: str = "",
                 data_requirements: List[str] = None,
                 **kwargs):
        """
        初始化因子元数据
        
        Args:
            name: 因子名称
            category: 因子类别
            description: 因子描述
            data_requirements: 数据需求
            **kwargs: 其他元数据
        """
        self.name = name
        
        # 处理类别
        if isinstance(category, FactorCategory):
            self.category = category
        else:
            try:
                self.category = FactorCategory[category.upper()]
            except (KeyError, AttributeError):
                self.category = FactorCategory.CUSTOM
                raise ValueError(f"未知的因子类别 '{category}'，使用默认类别 'CUSTOM'")
        
        self.description = description
        self.data_requirements = data_requirements or []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.test_results = {}
        
        # 添加其他元数据
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'category': self.category.name,
            'description': self.description,
            'data_requirements': self.data_requirements,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'test_results': self.test_results,
            # 添加其他属性
            **{k: v for k, v in self.__dict__.items() 
               if k not in ['name', 'category', 'description', 'data_requirements', 
                          'created_at', 'updated_at', 'test_results']}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorMetadata':
        """从字典创建"""
        category = data.pop('category')
        name = data.pop('name')
        description = data.pop('description', "")
        data_requirements = data.pop('data_requirements', [])
        
        # 创建实例
        instance = cls(
            name=name,
            category=category,
            description=description,
            data_requirements=data_requirements
        )
        
        # 添加其他属性
        for key, value in data.items():
            setattr(instance, key, value)
        
        return instance


class FactorRegistry:
    """因子注册表类"""
    
    def __init__(self, registry_path: str = "factor_registry.json"):
        """
        初始化因子注册表
        
        Args:
            registry_path: 注册表文件路径
        """
        self.registry_path = Path(registry_path)
        self.registry = {}
        
        # 加载现有注册表
        self._load_registry()
    
    def _load_registry(self):
        """加载注册表"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for name, factor_data in data.items():
                    self.registry[name] = FactorMetadata.from_dict(factor_data)
                
                logger.info(f"从 {self.registry_path} 加载了 {len(self.registry)} 个因子")
            except Exception as e:
                logger.error(f"加载注册表失败: {e}")
                self.registry = {}
    
    def _save_registry(self):
        """保存注册表"""
        # 确保目录存在
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 转换为可序列化的字典
            serializable_data = {
                name: metadata.to_dict() 
                for name, metadata in self.registry.items()
            }
            
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"注册表已保存到 {self.registry_path}")
        except Exception as e:
            logger.error(f"保存注册表失败: {e}")
    
    def register_factor(self, 
                       name: str,
                       category: Union[str, FactorCategory],
                       description: str = "",
                       data_requirements: List[str] = None,
                       **kwargs) -> bool:
        """
        注册因子
        
        Args:
            name: 因子名称
            category: 因子类别
            description: 因子描述
            data_requirements: 数据需求
            **kwargs: 其他元数据
            
        Returns:
            是否注册成功
        """
        if name in self.registry:
            raise ValueError(f"因子 '{name}' 已存在，更新元数据")
            # 更新现有因子
            metadata = self.registry[name]
            metadata.category = category if isinstance(category, FactorCategory) else \
                                FactorCategory[category.upper()] if category.upper() in FactorCategory.__members__ else \
                                FactorCategory.CUSTOM
            metadata.description = description
            if data_requirements:
                metadata.data_requirements = data_requirements
            metadata.updated_at = datetime.now().isoformat()
            
            # 更新其他元数据
            for key, value in kwargs.items():
                setattr(metadata, key, value)
        else:
            # 创建新因子
            self.registry[name] = FactorMetadata(
                name=name,
                category=category,
                description=description,
                data_requirements=data_requirements,
                **kwargs
            )
            logger.info(f"注册新因子: {name}")
        
        # 保存注册表
        self._save_registry()
        return True
    
    def get_factor(self, name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self.registry.get(name)
    
    def list_factors(self, 
                    category: Union[str, FactorCategory] = None) -> List[str]:
        """
        列出因子
        
        Args:
            category: 筛选的因子类别
            
        Returns:
            因子名称列表
        """
        if category is None:
            return list(self.registry.keys())
        
        # 处理类别
        if isinstance(category, str):
            try:
                category = FactorCategory[category.upper()]
            except KeyError:
                raise ValueError(f"未知的因子类别 '{category}'")

        
        # 按类别筛选
        return [
            name for name, metadata in self.registry.items()
            if metadata.category == category
        ]
    
    def get_factor_summary(self) -> pd.DataFrame:
        """获取因子摘要"""
        data = []
        
        for name, metadata in self.registry.items():
            row = {
                'name': name,
                'category': metadata.category.value,
                'description': metadata.description,
                'created_at': metadata.created_at,
                'updated_at': metadata.updated_at
            }
            
            # 添加测试结果
            if hasattr(metadata, 'test_results') and metadata.test_results:
                for key, value in metadata.test_results.items():
                    row[f'test_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def update_factor_test_result(self, 
                                name: str, 
                                test_results: Dict[str, Any]) -> bool:
        """
        更新因子测试结果
        
        Args:
            name: 因子名称
            test_results: 测试结果
            
        Returns:
            是否更新成功
        """
        if name not in self.registry:
            raise ValueError(f"因子 '{name}' 不存在，无法更新测试结果")

        
        # 更新测试结果
        metadata = self.registry[name]
        metadata.test_results.update(test_results)
        metadata.updated_at = datetime.now().isoformat()
        
        # 保存注册表
        self._save_registry()
        return True
    
    def delete_factor(self, name: str) -> bool:
        """删除因子"""
        if name not in self.registry:
            raise ValueError(f"因子 '{name}' 不存在，无法删除")

        
        # 删除因子
        del self.registry[name]
        logger.info(f"删除因子: {name}")
        
        # 保存注册表
        self._save_registry()
        return True 