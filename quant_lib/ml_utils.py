"""
机器学习工具模块

提供机器学习模型训练、评估和预测功能，支持特征工程、模型选择和超参数优化。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
from sklearn.model_selection import (
    train_test_split, 
    TimeSeriesSplit, 
    GridSearchCV, 
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import os
import datetime

# 获取模块级别的logger
logger = logging.getLogger(__name__)


class FeatureProcessor:
    """特征处理类"""
    
    def __init__(self):
        """初始化特征处理器"""
        self.scalers = {}
        self.feature_names = []
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     scaling: str = 'standard',
                     remove_outliers: bool = True,
                     fill_missing: str = 'mean') -> pd.DataFrame:
        """
        拟合并转换特征
        
        Args:
            X: 特征DataFrame
            scaling: 缩放方法，'standard'或'minmax'
            remove_outliers: 是否去除异常值
            fill_missing: 填充缺失值的方法，'mean'、'median'或'zero'
            
        Returns:
            处理后的特征DataFrame
        """
        logger.info("开始特征处理...")
        
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 处理缺失值
        X_processed = self._fill_missing_values(X, method=fill_missing)
        
        # 去除异常值
        if remove_outliers:
            X_processed = self._remove_outliers(X_processed)
        
        # 特征缩放
        X_scaled = self._scale_features(X_processed, method=scaling)
        
        logger.info("特征处理完成")
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换特征
        
        Args:
            X: 特征DataFrame
            
        Returns:
            处理后的特征DataFrame
        """
        # 检查特征名称是否一致
        if not all(col in X.columns for col in self.feature_names):
            logger.warning("输入特征与训练特征不一致")
            X = X[self.feature_names]
        
        # 转换特征
        X_transformed = X.copy()
        
        for col, scaler in self.scalers.items():
            if col in X_transformed.columns:
                X_transformed[col] = scaler.transform(X_transformed[[col]])
        
        return X_transformed
    
    def _fill_missing_values(self, X: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """填充缺失值"""
        X_filled = X.copy()
        
        for col in X_filled.columns:
            if X_filled[col].isna().sum() > 0:
                if method == 'mean':
                    X_filled[col] = X_filled[col].fillna(X_filled[col].mean())
                elif method == 'median':
                    X_filled[col] = X_filled[col].fillna(X_filled[col].median())
                elif method == 'zero':
                    X_filled[col] = X_filled[col].fillna(0)
                else:
                    logger.warning(f"不支持的填充方法: {method}，使用均值填充")
                    X_filled[col] = X_filled[col].fillna(X_filled[col].mean())
        
        return X_filled
    
    def _remove_outliers(self, X: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
        """去除异常值"""
        X_clean = X.copy()
        
        for col in X_clean.columns:
            mean = X_clean[col].mean()
            std = X_clean[col].std()
            
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            
            # 将异常值替换为边界值
            X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
        
        return X_clean
    
    def _scale_features(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """特征缩放"""
        X_scaled = X.copy()
        
        for col in X_scaled.columns:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                logger.warning(f"不支持的缩放方法: {method}，使用标准化缩放")
                scaler = StandardScaler()
            
            X_scaled[col] = scaler.fit_transform(X_scaled[[col]])
            self.scalers[col] = scaler
        
        return X_scaled


class ModelTrainer:
    """模型训练类"""
    
    def __init__(self, 
                model_type: str = 'lightgbm',
                task_type: str = 'regression',
                random_state: int = 42):
        """
        初始化模型训练器
        
        Args:
            model_type: 模型类型，支持'lightgbm', 'xgboost', 'random_forest', 'linear', 'svm'
            task_type: 任务类型，'regression'或'classification'
            random_state: 随机种子
        """
        self.model_type = model_type
        self.task_type = task_type
        self.random_state = random_state
        self.model = None
        self.feature_processor = FeatureProcessor()
        self.feature_importance = None
        self.best_params = None
    
    def train(self, 
             X: pd.DataFrame, 
             y: pd.Series,
             test_size: float = 0.2,
             cv: int = 5,
             tune_hyperparams: bool = True,
             param_grid: Optional[Dict] = None) -> Dict:
        """
        训练模型
        
        Args:
            X: 特征DataFrame
            y: 目标变量Series
            test_size: 测试集比例
            cv: 交叉验证折数
            tune_hyperparams: 是否调优超参数
            param_grid: 超参数网格
            
        Returns:
            评估指标字典
        """
        logger.info(f"开始训练{self.model_type}模型...")
        
        # 特征处理
        X_processed = self.feature_processor.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=self.random_state
        )
        
        # 创建模型
        self.model = self._create_model()
        
        # 超参数调优
        if tune_hyperparams:
            self._tune_hyperparameters(X_train, y_train, cv, param_grid)
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        metrics = self._evaluate_model(X_test, y_test)
        
        # 获取特征重要性
        self.feature_importance = self._get_feature_importance(X.columns)
        
        logger.info(f"模型训练完成，测试集评估指标: {metrics}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征DataFrame
            
        Returns:
            预测结果
        """
        if self.model is None:
            logger.error("模型尚未训练")
            return np.array([])
        
        # 特征处理
        X_processed = self.feature_processor.transform(X)
        
        # 预测
        if self.task_type == 'regression':
            return self.model.predict(X_processed)
        else:  # classification
            return self.model.predict_proba(X_processed)[:, 1]
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            logger.error("模型尚未训练，无法保存")
            return
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        joblib.dump({
            'model': self.model,
            'feature_processor': self.feature_processor,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'model_type': self.model_type,
            'task_type': self.task_type
        }, path)
        
        logger.info(f"模型已保存至: {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        if not os.path.exists(path):
            logger.error(f"模型文件不存在: {path}")
            return
        
        # 加载模型
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_processor = model_data['feature_processor']
        self.feature_importance = model_data['feature_importance']
        self.best_params = model_data['best_params']
        self.model_type = model_data['model_type']
        self.task_type = model_data['task_type']
        
        logger.info(f"模型已从 {path} 加载")
    
    def plot_feature_importance(self, figsize: Tuple[int, int] = (12, 6)):
        """
        绘制特征重要性图
        
        Args:
            figsize: 图表大小
        """
        if self.feature_importance is None:
            logger.error("特征重要性尚未计算")
            return
        
        plt.figure(figsize=figsize)
        
        # 按重要性排序
        sorted_idx = np.argsort(self.feature_importance)
        sorted_features = [self.feature_processor.feature_names[i] for i in sorted_idx]
        
        # 绘制条形图
        plt.barh(sorted_features, self.feature_importance[sorted_idx])
        plt.xlabel('特征重要性')
        plt.title('特征重要性排序')
        plt.tight_layout()
        plt.show()
    
    def _create_model(self):
        """创建模型"""
        if self.task_type == 'regression':
            if self.model_type == 'lightgbm':
                return lgb.LGBMRegressor(random_state=self.random_state)
            elif self.model_type == 'xgboost':
                return xgb.XGBRegressor(random_state=self.random_state)
            elif self.model_type == 'random_forest':
                return RandomForestRegressor(random_state=self.random_state)
            elif self.model_type == 'linear':
                return LinearRegression()
            elif self.model_type == 'svm':
                return SVR()
            else:
                logger.warning(f"不支持的模型类型: {self.model_type}，使用LightGBM")
                return lgb.LGBMRegressor(random_state=self.random_state)
        else:  # classification
            if self.model_type == 'lightgbm':
                return lgb.LGBMClassifier(random_state=self.random_state)
            elif self.model_type == 'xgboost':
                return xgb.XGBClassifier(random_state=self.random_state)
            elif self.model_type == 'random_forest':
                return RandomForestClassifier(random_state=self.random_state)
            elif self.model_type == 'linear':
                return LogisticRegression(random_state=self.random_state)
            elif self.model_type == 'svm':
                return SVC(probability=True, random_state=self.random_state)
            else:
                logger.warning(f"不支持的模型类型: {self.model_type}，使用LightGBM")
                return lgb.LGBMClassifier(random_state=self.random_state)
    
    def _tune_hyperparameters(self, 
                             X_train: pd.DataFrame, 
                             y_train: pd.Series,
                             cv: int,
                             param_grid: Optional[Dict] = None):
        """超参数调优"""
        logger.info("开始超参数调优...")
        
        # 默认参数网格
        if param_grid is None:
            if self.model_type == 'lightgbm':
                param_grid = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 63, 127]
                }
            elif self.model_type == 'xgboost':
                param_grid = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5]
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'linear':
                if self.task_type == 'regression':
                    param_grid = {}  # 线性回归没有需要调优的超参数
                else:
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'penalty': ['l1', 'l2']
                    }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.1, 1]
                }
        
        # 创建交叉验证对象
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # 创建网格搜索对象
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error' if self.task_type == 'regression' else 'roc_auc',
            n_jobs=-1
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        # 更新模型
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"最优超参数: {self.best_params}")
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """评估模型"""
        # 预测
        if self.task_type == 'regression':
            y_pred = self.model.predict(X_test)
            
            # 计算评估指标
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        else:  # classification
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = self.model.predict(X_test)
            
            # 计算评估指标
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
        
        return metrics
    
    def _get_feature_importance(self, feature_names: List[str]) -> np.ndarray:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            logger.warning(f"{self.model_type}模型不支持特征重要性")
            return np.zeros(len(feature_names))


class TimeSeriesFeatureGenerator:
    """时间序列特征生成器"""
    
    def __init__(self):
        """初始化时间序列特征生成器"""
        pass
    
    def generate_features(self, 
                         df: pd.DataFrame, 
                         target_col: str,
                         lag_periods: List[int] = [1, 5, 10, 20],
                         rolling_windows: List[int] = [5, 10, 20, 60],
                         date_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        生成时间序列特征
        
        Args:
            df: 输入DataFrame，index为日期
            target_col: 目标列名
            lag_periods: 滞后期列表
            rolling_windows: 滚动窗口列表
            date_features: 是否生成日期特征
            
        Returns:
            (特征DataFrame, 目标变量Series)
        """
        logger.info("开始生成时间序列特征...")
        
        # 复制数据
        data = df.copy()
        
        # 确保索引是日期类型
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                logger.error("索引无法转换为日期类型")
                return pd.DataFrame(), pd.Series()
        
        # 目标变量
        y = data[target_col].shift(-1)  # 预测下一期的值
        
        # 生成特征
        X = pd.DataFrame(index=data.index)
        
        # 添加原始特征
        for col in data.columns:
            if col != target_col:
                X[col] = data[col]
        
        # 生成滞后特征
        for lag in lag_periods:
            X[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # 生成滚动特征
        for window in rolling_windows:
            X[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
            X[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window=window).std()
            X[f'{target_col}_rolling_min_{window}'] = data[target_col].rolling(window=window).min()
            X[f'{target_col}_rolling_max_{window}'] = data[target_col].rolling(window=window).max()
        
        # 生成日期特征
        if date_features:
            X['day_of_week'] = data.index.dayofweek
            X['day_of_month'] = data.index.day
            X['month'] = data.index.month
            X['quarter'] = data.index.quarter
            X['year'] = data.index.year
            X['is_month_end'] = data.index.is_month_end.astype(int)
            X['is_quarter_end'] = data.index.is_quarter_end.astype(int)
        
        # 删除缺失值
        X = X.dropna()
        y = y.loc[X.index]
        
        logger.info(f"特征生成完成，共 {X.shape[1]} 个特征")
        return X, y


# 工厂函数，方便创建模型训练器
def create_model_trainer(
    model_type: str = 'lightgbm',
    task_type: str = 'regression',
    random_state: int = 42
) -> ModelTrainer:
    """
    创建模型训练器
    
    Args:
        model_type: 模型类型
        task_type: 任务类型
        random_state: 随机种子
        
    Returns:
        ModelTrainer实例
    """
    return ModelTrainer(
        model_type=model_type,
        task_type=task_type,
        random_state=random_state
    )