"""
滚动IC管理器 - 解决前视偏差的关键组件

核心功能：
1. 时点化IC计算：严格按时间点滚动计算IC，避免未来信息泄露
2. 增量存储：支持增量计算和存储，提升效率
3. 窗口管理：灵活的回看窗口配置
4. 数据完整性：确保IC计算的时间一致性

设计理念：
- 完全杜绝前视偏差
- 支持实盘级别的严格时间控制
- 高效的增量计算和存储
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import logging

from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from projects._03_factor_selection.factory.config_snapshot_manager import ConfigSnapshotManager
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class ICCalculationConfig:
    """IC计算配置"""
    lookback_months: int = 12  # 回看窗口(月) 目前写死-注意调整 0.1
    forward_periods: List[str] = None  # 前向收益周期
    min_observations: int = 120  # 最小观测数量  目前写死-注意调整 0.1
    calculation_frequency: str = 'M'  # 计算频率 ('M'=月末, 'Q'=季末)

    def __init__(self,lookback_months=12, forward_periods: list=None , min_observations: int = 120, calculation_frequency: str = 'M',calcu_type='c2c', version='20190328_20231231'):
        self.lookback_months = lookback_months
        self.forward_periods = forward_periods
        self.min_observations = min_observations
        self.calculation_frequency = calculation_frequency

        self.calcu_type=calcu_type
        self.version=version


@dataclass
class ICSnapshot:
    """IC快照数据结构"""
    calculation_date: str  # 计算时点
    factor_name: str  # 因子名称
    stock_pool_index: str  # 股票池
    window_start: str  # 回看窗口起点
    window_end: str  # 回看窗口终点
    ic_stats: Dict[str, Dict]  # 各周期IC统计
    metadata: Dict  # 元数据信息


class RollingICManager:
    """滚动IC管理器 - 无前视偏差的IC计算与存储"""

    def __init__(self,calcu_return_type, config: Optional[ICCalculationConfig] = None,version=None):
        self.main_work_path = Path(r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace\result")
        self.config = config or ICCalculationConfig()
        self.calcu_return_type=calcu_return_type
        self.version = version

        # 时点IC索引
        self._ic_index = {}
        self._load_ic_index()
    #ok
    def calculate_and_store_rolling_ic(
            self,
            factor_names: List[str],
            stock_pool: str,
            start_date: str,
            end_date: str,
            resultLoadManager:ResultLoadManager,  # 数据源
            force_recalculate: bool = False
    ) -> Dict[str, List[ICSnapshot]]:
        """
        计算并存储滚动IC
        
        Args:
            factor_names: 因子名称列表
            stock_pool: 股票池名称
            start_date: 开始计算时点
            end_date: 结束计算时点
            factor_data_source: 因子数据源
            return_data_source: 收益数据源
            force_recalculate: 是否强制重新计算
            
        Returns:
            Dict[factor_name, List[ICSnapshot]]: 所有因子的IC快照序列
        """
        logger.info(f"🔄 开始滚动IC计算: {start_date} -> {end_date}")
        logger.info(f"📊 因子数量: {len(factor_names)}, 股票池: {stock_pool}")

        # 1. 生成计算时点序列
        calculation_dates = self._generate_calculation_dates(start_date, end_date)
        logger.info(f"⏰ 计算时点数量: {len(calculation_dates)}")

        # 2. 逐时点计算IC
        all_factor_snapshots = {name: [] for name in factor_names}

        for calc_date in calculation_dates:
            logger.info(f"📅 计算时点: {calc_date}")

            for factor_name in factor_names:
                try:
                    # 检查是否已存在计算结果
                    if not force_recalculate and self._snapshot_exists(
                            factor_name, stock_pool, calc_date
                    ):
                        snapshot = self._load_snapshot(factor_name, stock_pool, calc_date)
                        logger.debug(f"  📥 {factor_name}: 使用已有快照")
                    else:
                        # 计算新的IC快照
                        snapshot = self._calculate_ic_snapshot(
                            factor_name, stock_pool, calc_date,
                            resultLoadManager
                        )

                        if snapshot:
                            self._save_snapshot(snapshot)
                            logger.debug(f"  ✅ {factor_name}:{calc_date} IC快照计算完成")
                        else:#很正常啊，比如不满足观测点个数的时候
                            continue

                    all_factor_snapshots[factor_name].append(snapshot)

                except Exception as e:
                    logger.error(f"  ❌ {factor_name}: IC计算异常 - {e}")
                    continue

        logger.info(f"✅ 滚动IC计算完成")
        return all_factor_snapshots

    def get_ic_at_timepoint(
            self,
            factor_name: str,
            stock_pool: str,
            calculation_date: str
    ) -> Optional[ICSnapshot]:
        """获取指定时点的IC快照"""
        return self._load_snapshot(factor_name, stock_pool, calculation_date)

    def get_ic_series(
            self,
            factor_name: str,
            stock_pool: str,
            start_date: str,
            end_date: str
    ) -> List[ICSnapshot]:
        """获取时间序列的IC快照"""
        snapshots = []

        # 从索引中查找符合条件的快照
        key_pattern = f"{factor_name}_{stock_pool}"

        for key, metadata in self._ic_index.items():
            if key.startswith(key_pattern):
                calc_date = metadata['calculation_date']
                if start_date <= calc_date <= end_date:
                    snapshot = self._load_snapshot(factor_name, stock_pool, calc_date)
                    if snapshot:
                        snapshots.append(snapshot)

        # 按计算时点排序
        snapshots.sort(key=lambda x: x.calculation_date)
        return snapshots

    def _calculate_ic_snapshot(
            self,
            factor_name: str,
            stock_pool_index: str,
            calculation_date: str,
            resultLoadManager:ResultLoadManager
    ) -> Optional[ICSnapshot]:
        """计算单个时点的IC快照"""
        try:
            # 1. 确定回看窗口（严格避免前视偏差）
            calc_date = pd.Timestamp(calculation_date)
            window_end = calc_date
            window_start = calc_date - relativedelta(months=self.config.lookback_months)

            # 2. 获取窗口内的因子数据
            factor_data = resultLoadManager.get_factor_data(
                factor_name, stock_pool_index,
                window_start.strftime('%Y-%m-%d'),
                window_end.strftime('%Y-%m-%d')
            )

            if factor_data is None or factor_data.empty:
                raise ValueError(f"因子 {factor_name} 在窗口 {window_start}-{window_end} 内无数据")

            # 3. 计算各周期IC统计
            ic_stats = {}

            for period in self.config.forward_periods:
                # 获取前向收益数据
                period_days = period
                return_end = window_end + timedelta(days=period_days + 10)  # 留充足余量

                return_data = resultLoadManager.get_return_data(
                    stock_pool_index,
                    window_start.strftime('%Y-%m-%d'),
                    return_end.strftime('%Y-%m-%d'),
                    period_days
                )

                if return_data is None or return_data.empty:
                    raise ValueError('收益率数据不可能为空！，严重错误！')

                # 计算IC
                period_ic_stats = self._calculate_period_ic(factor_data, return_data)

                if period_ic_stats:
                    ic_stats[period] = period_ic_stats

            if not ic_stats:
                logger.warning(f"因子 {factor_name} 在时点 {calculation_date} 无有效IC统计--正常：因为不满足120个观测点！（人话：回头看的天数没有达到120天")
                return None

            # 4. 构建IC快照
            snapshot = ICSnapshot(
                calculation_date=calculation_date,
                factor_name=factor_name,
                stock_pool_index=stock_pool_index,
                window_start=window_start.strftime('%Y-%m-%d'),
                window_end=window_end.strftime('%Y-%m-%d'),
                ic_stats=ic_stats,
                metadata={
                    'config': {
                        'lookback_months': self.config.lookback_months,
                        'min_observations': self.config.min_observations
                    },
                    'data_points': len(factor_data),
                    'created_timestamp': datetime.now().isoformat()
                }
            )

            return snapshot

        except Exception as e:
            logger.error(f"计算IC快照失败 {factor_name}@{calculation_date}: {e}")
            return None

    def _calculate_period_ic(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> Optional[Dict]:
        """计算特定周期的IC统计"""
        try:
            # 对齐因子和收益数据
            aligned_factor, aligned_return = self._align_data(factor_data, return_data)

            if len(aligned_factor) < self.config.min_observations:
                return None

            # 计算IC序列
            ic_series = aligned_factor.corrwith(aligned_return, axis=1)
            ic_series = ic_series.dropna()

            if len(ic_series) == 0:
                raise ValueError("corrwith之后IC序列为空")

            # IC统计指标
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_win_rate = (ic_series > 0).mean()

            # t检验
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(ic_series, 0)

            return {
                'ic_mean': float(ic_mean),
                'ic_std': float(ic_std),
                'ic_ir': float(ic_ir),
                'ic_win_rate': float(ic_win_rate),
                'ic_t_stat': float(t_stat),
                'ic_p_value': float(p_value),
                'ic_count': len(ic_series),
                'ic_max': float(ic_series.max()),
                'ic_min': float(ic_series.min())
            }

        except Exception as e:
            raise  ValueError(f"计算周期IC失败: {e}")

    def _align_data(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """对齐因子和收益数据"""
        # 找到共同的时间和股票
        common_dates = factor_data.index.intersection(return_data.index)
        common_stocks = factor_data.columns.intersection(return_data.columns)

        aligned_factor = factor_data.loc[common_dates, common_stocks]
        aligned_return = return_data.loc[common_dates, common_stocks]

        return aligned_factor, aligned_return

    def _generate_calculation_dates(self, start_date: str, end_date: str) -> List[str]:
        """生成计算时点序列"""
        dates = []
        current = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # 根据频率生成时点
        if self.config.calculation_frequency == 'M':
            # 月末
            while current <= end:
                # 找到当月最后一个工作日
                month_end = current + pd.offsets.MonthEnd(0)
                if month_end <= end:
                    dates.append(month_end.strftime('%Y-%m-%d'))
                current = current + pd.offsets.MonthEnd(1)
        elif self.config.calculation_frequency == 'Q':
            # 季末
            while current <= end:
                quarter_end = current + pd.offsets.QuarterEnd(0)
                if quarter_end <= end:
                    dates.append(quarter_end.strftime('%Y-%m-%d'))
                current = current + pd.offsets.QuarterEnd(1)

        return dates

    def _snapshot_exists(self, factor_name: str, stock_pool: str, calculation_date: str) -> bool:
        """检查IC快照是否已存在"""
        snapshot_key = f"{factor_name}_{stock_pool}_{calculation_date}"
        return snapshot_key in self._ic_index

    def _save_snapshot(self, snapshot: ICSnapshot):
        """保存IC快照"""
        # 构建文件路径
        snapshot_dir = self.main_work_path / snapshot.stock_pool_index / snapshot.factor_name / self.calcu_return_type / self.version / 'rolling_ic'
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        filename = f"ic_snapshot_{snapshot.calculation_date}.json"
        filepath = snapshot_dir / filename

        # 序列化快照
        snapshot_dict = {
            'calculation_date': snapshot.calculation_date,
            'factor_name': snapshot.factor_name,
            'stock_pool_index': snapshot.stock_pool_index,
            'window_start': snapshot.window_start,
            'window_end': snapshot.window_end,
            'ic_stats': snapshot.ic_stats,
            'metadata': snapshot.metadata
        }

        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(snapshot_dict, f, ensure_ascii=False, indent=2)

        # 更新索引
        snapshot_key = f"{snapshot.factor_name}_{snapshot.stock_pool_index}_version_{self.version}_calculation_date_{snapshot.calculation_date}"
        self._ic_index[snapshot_key] = {
            'calculation_date': snapshot.calculation_date,
            'filepath': str(filepath),
            'created_at': datetime.now().isoformat()
        }

        self._save_ic_index()
        logger.debug(f"IC快照已保存: {filepath}")

    def _load_snapshot(self, factor_name: str, stock_pool_index: str, calculation_date: str) -> Optional[ICSnapshot]:
        """加载IC快照"""
        snapshot_key =f'{factor_name}_{stock_pool_index}_version_{self.version}_calculation_date_{calculation_date}'

        if snapshot_key not in self._ic_index:
            return None

        try:
            filepath = self._ic_index[snapshot_key]['filepath']

            with open(filepath, 'r', encoding='utf-8') as f:
                snapshot_dict = json.load(f)

            return ICSnapshot(**snapshot_dict)

        except Exception as e:
            logger.error(f"加载IC快照失败 {snapshot_key}: {e}")
            return None

    def _load_ic_index(self):
        """加载IC索引"""
        index_file = self.main_work_path / "ic_index.json"

        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    self._ic_index = json.load(f)
                logger.info(f"IC索引加载完成，共 {len(self._ic_index)} 条记录")
            except Exception as e:
                raise ValueError(f"加载IC索引失败: {e}")
        else:
            self._ic_index = {}

    def _save_ic_index(self):
        """保存IC索引"""
        index_file = self.main_work_path / "ic_index.json"

        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self._ic_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存IC索引失败: {e}")

    def get_latest_calculation_date(self, factor_name: str, stock_pool: str) -> Optional[str]:
        """获取因子的最新计算时点"""
        pattern = f"{factor_name}_{stock_pool}_"
        latest_date = None

        for key, metadata in self._ic_index.items():
            if key.startswith(pattern):
                calc_date = metadata['calculation_date']
                if latest_date is None or calc_date > latest_date:
                    latest_date = calc_date

        return latest_date

    def cleanup_old_snapshots(self, keep_months: int = 36):
        """清理过期的IC快照"""
        cutoff_date = datetime.now() - relativedelta(months=keep_months)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        removed_count = 0
        keys_to_remove = []

        for key, metadata in self._ic_index.items():
            if metadata['calculation_date'] < cutoff_str:
                try:
                    # 删除文件
                    filepath = Path(metadata['filepath'])
                    if filepath.exists():
                        filepath.unlink()

                    keys_to_remove.append(key)
                    removed_count += 1

                except Exception as e:
                    logger.error(f"删除快照失败 {key}: {e}")

        # 更新索引
        for key in keys_to_remove:
            del self._ic_index[key]

        self._save_ic_index()
        logger.info(f"清理完成，删除 {removed_count} 个过期快照")


def run_cal_and_save_rolling_ic_by_snapshot_config_id(snapshot_config,factor_names ):
    manager = ConfigSnapshotManager()
    pool_index,s,e ,config_evaluation= manager.get_snapshot_config_content_details(snapshot_config)
    version = f'{s}_{e}'
    config = ICCalculationConfig(
        lookback_months=12,
        forward_periods=config_evaluation['forward_periods'],
        min_observations=1,
        calculation_frequency='M'
    )
    if len(config_evaluation['returns_calculator']) > 1:
        raise ValueError("目前只支持一个返回计算器 请指定 ")
    calcu_return_type=config_evaluation['returns_calculator'][0]
    manager = RollingICManager(calcu_return_type, config,version)

    resultLoadManager = ResultLoadManager(calcu_return_type=calcu_return_type, version=version,
                                          core_eveluation_type='ic', is_raw_factor=False)

    stock_pool_index = pool_index

    snapshots = manager.calculate_and_store_rolling_ic(
        factor_names, stock_pool_index, s, e,
        resultLoadManager
    )
    print(f"计算完成，共生成 {sum(len(snaps) for snaps in snapshots.values())} 个IC快照")
if __name__ == '__main__':
    run_cal_and_save_rolling_ic_by_snapshot_config_id('20250825_091622_98ed2d09',factor_names = ['log_circ_mv'])