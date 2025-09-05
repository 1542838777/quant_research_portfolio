"""
Backtrader配置管理器

功能：
1. 无缝兼容现有的BacktestConfig配置
2. 提供一键迁移的配置转换
3. 预设多种常用的策略配置模板
4. 配置验证和优化建议
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any
import pandas as pd
from pathlib import Path

from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class BacktraderConfig:
    """
    Backtrader专用配置类 - 扩展原有BacktestConfig
    
    新增功能：
    - 重试机制配置
    - 动态权重管理
    - 增强的风控参数
    - 调试和监控配置
    """
    
    # === 基础策略参数（兼容原有BacktestConfig）===
    top_quantile: float = 0.2
    rebalancing_freq: str = 'M'
    max_positions: int = 10
    max_holding_days: int = 60
    
    # === 交易成本参数（兼容原有）===
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    stamp_duty: float = 0.001
    min_commission: float = 5.0
    initial_cash: float = 1000000.0
    
    # === 新增：Backtrader特有参数 ===
    retry_buy_days: int = 3              # 买入重试天数
    retry_sell_days: int = 50             # 卖出重试天数
    enable_forced_exits: bool = True     # 启用强制卖出
    #  设置冷却期参数，例如卖出后10个交易日内不允许再买入
    buy_after_sell_cooldown = 10
    enable_retry_mechanism: bool = True   # 启用重试机制
    trading_days: list = None         #  交易日期列表
    real_wide_close_price: pd.DataFrame =None        #  真实价格
    _buy_success_num: dict=None
    _sell_success_num: dict=None

    # === 新增：动态权重管理 ===
    use_dynamic_weights: bool = True     # 使用动态权重分配
    max_weight_per_stock: float = 0.15   # 单股最大权重
    min_weight_threshold: float = 0.01   # 最小权重阈值
    weight_rebalance_tolerance: float = 0.05  # 权重再平衡容忍度
    
    # === 新增：增强风控参数 ===
    emergency_exit_threshold: float = -0.2    # 紧急止损阈值(-20%)
    max_daily_trades: int = 20                # 单日最大交易笔数
    min_cash_reserve: float = 0.02            # 最小现金储备比例
    
    # === 新增：调试和监控配置 ===
    debug_mode: bool = True                   # 调试模式
    log_detailed_trades: bool = False         # 详细交易日志
    log_failed_orders: bool = True            # 记录失败订单
    save_daily_stats: bool = False            # 保存每日统计
    
    # === 新增：数据质量控制 ===
    min_data_coverage: float = 0.8           # 最小数据覆盖率
    max_missing_consecutive_days: int = 5     # 最大连续缺失天数
    min_trading_days: int = 250              # 最小交易天数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def get_comprehensive_fee_rate(self) -> float:
        """
        计算综合费率 - 与原有vectorBT逻辑保持一致
        
        Returns:
            float: 综合费率
        """
        return (
            self.commission_rate +      # 佣金
            self.slippage_rate +        # 滑点
            self.stamp_duty / 2         # 印花税分摊
        )
    
    def validate(self) -> Dict[str, List[str]]:
        """
        配置验证 - 检查参数合理性
        
        Returns:
            Dict: 验证结果 {'errors': [], 'warnings': []}
        """
        errors = []
        warnings = []
        
        # 基础参数检查
        if not 0 < self.top_quantile <= 1:
            errors.append(f"top_quantile必须在(0,1]范围内，当前值: {self.top_quantile}")
        
        if self.max_positions <= 0:
            errors.append(f"max_positions必须大于0，当前值: {self.max_positions}")
        
        if self.initial_cash <= 0:
            errors.append(f"initial_cash必须大于0，当前值: {self.initial_cash}")
        
        # 费率检查
        total_fee_rate = self.get_comprehensive_fee_rate()
        if total_fee_rate > 0.01:  # 1%
            warnings.append(f"综合费率过高: {total_fee_rate:.4f} (1%)")
        
        # 风控参数检查
        if self.max_weight_per_stock > 1.0 / self.max_positions:
            warnings.append(f"单股最大权重({self.max_weight_per_stock:.2%})可能导致集中度过高")
        
        # 重试参数检查
        if self.retry_buy_days > 10:
            warnings.append(f"买入重试期过长: {self.retry_buy_days}天")
        
        return {'errors': errors, 'warnings': warnings}
    
    def optimize_for_scenario(self, scenario: str) -> 'BacktraderConfig':
        """
        针对特定场景优化配置
        
        Args:
            scenario: 场景类型 ('conservative', 'aggressive', 'high_turnover', 'low_turnover')
            
        Returns:
            BacktraderConfig: 优化后的配置
        """
        optimized = BacktraderConfig(**self.to_dict())
        
        if scenario == 'conservative':
            # 保守配置：低换手，高质量
            optimized.top_quantile = 0.1           # 只买最好的10%
            optimized.rebalancing_freq = 'Q'       # 季度调仓
            optimized.max_holding_days = 120       # 长期持有
            optimized.max_positions = 5            # 集中持仓
            optimized.retry_buy_days = 5           # 更长重试期
            
        elif scenario == 'aggressive':
            # 激进配置：高换手，广撒网
            optimized.top_quantile = 0.3           # 买入30%
            optimized.rebalancing_freq = 'W'       # 周度调仓
            optimized.max_holding_days = 30        # 短期持有
            optimized.max_positions = 20           # 分散持仓
            optimized.retry_buy_days = 1           # 短重试期
            
        elif scenario == 'high_liquidity':
            # 高流动性配置：适合大资金
            optimized.max_weight_per_stock = 0.05  # 更低的单股权重
            optimized.min_cash_reserve = 0.05      # 更高现金储备
            optimized.max_daily_trades = 50        # 允许更多交易
            
        elif scenario == 'small_cap':
            # 小盘股配置：处理流动性问题
            optimized.slippage_rate = 0.003        # 更高滑点
            optimized.max_weight_per_stock = 0.08  # 适中的单股权重
            optimized.retry_buy_days = 7           # 更长的重试期
            optimized.min_data_coverage = 0.7      # 降低数据质量要求
        
        logger.info(f"配置已优化为{scenario}场景")
        return optimized


class ConfigMigrationHelper:
    """配置迁移助手"""
    
    @staticmethod
    def from_vectorbt_config(vectorbt_config) -> BacktraderConfig:
        """
        从vectorBT配置创建Backtrader配置
        
        Args:
            vectorbt_config: 原有的BacktestConfig对象
            
        Returns:
            BacktraderConfig: 转换后的配置
        """
        if vectorbt_config is None:
            return BacktraderConfig()
        
        # 提取所有兼容的参数
        compatible_params = {}
        vectorbt_fields = [
            'top_quantile', 'rebalancing_freq', 'max_positions', 'max_holding_days',
            'commission_rate', 'slippage_rate', 'stamp_duty', 'min_commission', 
            'initial_cash', 'max_weight_per_stock', 'min_weight_threshold'
        ]
        
        for field in vectorbt_fields:
            if hasattr(vectorbt_config, field):
                compatible_params[field] = getattr(vectorbt_config, field)
        
        # 创建Backtrader配置，使用兼容参数覆盖默认值
        bt_config = BacktraderConfig(**compatible_params)
        
        logger.info("配置迁移完成:")
        logger.info(f"  迁移参数: {len(compatible_params)}个")
        
        return bt_config
    
    @staticmethod
    def batch_migrate_configs(config_dict: Dict[str, Any]) -> Dict[str, BacktraderConfig]:
        """
        批量迁移配置
        
        Args:
            config_dict: 配置字典 {配置名: 配置对象}
            
        Returns:
            Dict: 迁移后的配置字典
        """
        migrated_configs = {}
        
        for config_name, config_obj in config_dict.items():
            try:
                migrated_config = ConfigMigrationHelper.from_vectorbt_config(config_obj)
                migrated_configs[config_name] = migrated_config
                logger.info(f"配置{config_name}迁移成功")
                
            except Exception as e:
                logger.error(f"配置{config_name}迁移失败: {e}")
                migrated_configs[config_name] = None
        
        return migrated_configs


class StrategyTemplates:
    """预设策略模板"""
    
    @staticmethod
    def get_all_templates() -> Dict[str, BacktraderConfig]:
        """获取所有预设模板"""
        templates = {
            'conservative_value': StrategyTemplates.conservative_value_strategy(),
            'aggressive_momentum': StrategyTemplates.aggressive_momentum_strategy(), 
            'balanced_quality': StrategyTemplates.balanced_quality_strategy(),
            'high_frequency': StrategyTemplates.high_frequency_strategy(),
            'institutional_grade': StrategyTemplates.institutional_grade_strategy()
        }
        
        logger.info(f"可用策略模板: {list(templates.keys())}")
        return templates
    
    @staticmethod
    def conservative_value_strategy() -> BacktraderConfig:
        """保守价值策略模板"""
        return BacktraderConfig(
            top_quantile=0.15,                    # 精选前15%
            rebalancing_freq='Q',                 # 季度调仓
            max_positions=8,                      # 集中持仓
            max_holding_days=180,                 # 长期持有
            commission_rate=0.0002,               # 较低费率
            slippage_rate=0.0008,
            initial_cash=5000000,                 # 较大资金
            retry_buy_days=5,                     # 耐心等待
            max_weight_per_stock=0.2,             # 允许更高集中度
            enable_forced_exits=False,            # 不强制卖出
            debug_mode=False
        )
    
    @staticmethod
    def aggressive_momentum_strategy() -> BacktraderConfig:
        """激进动量策略模板"""
        return BacktraderConfig(
            top_quantile=0.25,                    # 做多25%
            rebalancing_freq='W',                 # 周度调仓
            max_positions=15,                     # 适度分散
            max_holding_days=30,                  # 短期持有
            commission_rate=0.0003,
            slippage_rate=0.0015,                 # 高换手对应高滑点
            initial_cash=1000000,
            retry_buy_days=2,                     # 快速重试
            max_weight_per_stock=0.1,             # 分散风险
            enable_forced_exits=True,             # 启用强制卖出
            debug_mode=True
        )
    
    @staticmethod
    def balanced_quality_strategy() -> BacktraderConfig:
        """平衡质量策略模板"""
        return BacktraderConfig(
            top_quantile=0.2,                     # 做多20%
            rebalancing_freq='M',                 # 月度调仓
            max_positions=12,                     # 平衡持仓
            max_holding_days=60,                  # 中期持有
            commission_rate=0.0003,
            slippage_rate=0.001,
            initial_cash=2000000,
            retry_buy_days=3,                     # 标准重试
            max_weight_per_stock=0.12,            # 适度集中
            enable_forced_exits=True,
            debug_mode=False
        )
    
    @staticmethod
    def high_frequency_strategy() -> BacktraderConfig:
        """高频策略模板"""
        return BacktraderConfig(
            top_quantile=0.3,                     # 做多30%
            rebalancing_freq='W',                 # 周度调仓
            max_positions=25,                     # 高度分散
            max_holding_days=14,                  # 极短持有
            commission_rate=0.0002,               # 优惠费率
            slippage_rate=0.0005,                 # 低滑点（假设高频优势）
            initial_cash=10000000,                # 大资金
            retry_buy_days=1,                     # 极短重试
            max_weight_per_stock=0.06,            # 高度分散
            max_daily_trades=100,                 # 允许高频交易
            enable_forced_exits=True,
            log_detailed_trades=True,             # 详细监控
            debug_mode=True
        )
    
    @staticmethod
    def institutional_grade_strategy() -> BacktraderConfig:
        """机构级策略模板"""
        return BacktraderConfig(
            top_quantile=0.18,                    # 精选18%
            rebalancing_freq='M',                 # 月度调仓
            max_positions=30,                     # 机构级分散
            max_holding_days=90,                  # 中长期持有
            commission_rate=0.0001,               # 机构优惠费率
            slippage_rate=0.0003,                 # 低滑点
            stamp_duty=0.0005,                    # 优惠印花税
            initial_cash=50000000,                # 大资金
            retry_buy_days=7,                     # 充分重试
            max_weight_per_stock=0.05,            # 严格分散
            min_cash_reserve=0.03,                # 现金储备
            emergency_exit_threshold=-0.15,       # 较严格止损
            enable_retry_mechanism=True,
            save_daily_stats=True,                # 机构级监控
            debug_mode=False
        )


class MigrationValidator:
    """迁移验证器"""
    
    @staticmethod
    def validate_migration_readiness(price_df: pd.DataFrame, factor_dict: Dict[str, pd.DataFrame],
                                   config: BacktraderConfig) -> Dict[str, Any]:
        """
        验证迁移准备情况
        
        Args:
            price_df: 价格数据
            factor_dict: 因子数据字典
            config: 配置对象
            
        Returns:
            Dict: 验证结果
        """
        logger.info("验证迁移准备情况...")
        
        validation_result = {
            'is_ready': True,
            'data_quality': {},
            'config_issues': {},
            'recommendations': []
        }
        
        # 1. 数据质量检查
        data_issues = MigrationValidator._check_data_quality(price_df, factor_dict, config)
        validation_result['data_quality'] = data_issues
        
        # 2. 配置检查
        config_validation = config.validate()
        validation_result['config_issues'] = config_validation
        
        # 3. 生成建议
        recommendations = MigrationValidator._generate_recommendations(data_issues, config_validation, config)
        validation_result['recommendations'] = recommendations
        
        # 4. 判断是否准备就绪
        if data_issues['critical_issues'] or config_validation['errors']:
            validation_result['is_ready'] = False
        
        return validation_result
    
    @staticmethod
    def _check_data_quality(price_df: pd.DataFrame, factor_dict: Dict[str, pd.DataFrame], 
                          config: BacktraderConfig) -> Dict:
        """检查数据质量"""
        issues = {
            'critical_issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # 价格数据检查
        price_coverage = (1 - price_df.isnull().sum().sum() / price_df.size)
        if price_coverage < config.min_data_coverage:
            issues['critical_issues'].append(f"价格数据覆盖率过低: {price_coverage:.1%}")
        
        # 因子数据检查
        for factor_name, factor_data in factor_dict.items():
            factor_coverage = (1 - factor_data.isnull().sum().sum() / factor_data.size)
            if factor_coverage < config.min_data_coverage:
                issues['warnings'].append(f"{factor_name}数据覆盖率过低: {factor_coverage:.1%}")
        
        # 交易天数检查
        trading_days = len(price_df.index)
        if trading_days < config.min_trading_days:
            issues['warnings'].append(f"交易天数偏少: {trading_days}天")
        
        issues['stats'] = {
            'price_coverage': price_coverage,
            'trading_days': trading_days,
            'stock_count': len(price_df.columns)
        }
        
        return issues
    
    @staticmethod
    def _generate_recommendations(data_issues: Dict, config_issues: Dict, config: BacktraderConfig) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于数据质量的建议
        if data_issues['critical_issues']:
            recommendations.append("建议清理数据或降低数据质量要求")
        
        # 基于配置的建议
        if config.max_positions > 20 and config.initial_cash < 2000000:
            recommendations.append("资金较少但持仓分散，建议减少max_positions或增加资金")
        
        if config.rebalancing_freq == 'W' and config.retry_buy_days > 3:
            recommendations.append("高频调仓配置了长重试期，可能导致策略混乱")
        
        # 性能优化建议
        if config.debug_mode and config.log_detailed_trades:
            recommendations.append("生产环境建议关闭详细日志以提高性能")
        
        return recommendations


def demo_migration_process():
    """演示完整的迁移流程"""
    logger.info("=" * 80)
    logger.info("📋 完整迁移流程演示")
    logger.info("=" * 80)
    
    # 1. 假设你有原有的vectorBT配置
    from projects._04backtesting.quant_backtester import BacktestConfig
    
    original_config = BacktestConfig(
        top_quantile=0.2,
        rebalancing_freq='M', 
        commission_rate=0.0003,
        slippage_rate=0.001,
        stamp_duty=0.001,
        initial_cash=1000000,
        max_positions=10,
        max_holding_days=60
    )
    
    logger.info("原始vectorBT配置:")
    logger.info(f"  调仓频率: {original_config.rebalancing_freq}")
    logger.info(f"  做多分位: {original_config.top_quantile}")
    logger.info(f"  最大持仓: {original_config.max_positions}")
    
    # 2. 迁移配置
    bt_config = ConfigMigrationHelper.from_vectorbt_config(original_config)
    
    logger.info("迁移后Backtrader配置:")
    logger.info(f"  基础参数保持一致")
    logger.info(f"  新增重试机制: {bt_config.retry_buy_days}天")
    logger.info(f"  新增动态权重: {bt_config.use_dynamic_weights}")
    
    # 3. 配置验证
    validation = bt_config.validate()
    
    if validation['errors']:
        logger.error("配置验证发现错误:")
        for error in validation['errors']:
            logger.error(f"  ❌ {error}")
    
    if validation['warnings']:
        logger.warning("配置验证发现警告:")
        for warning in validation['warnings']:
            logger.warning(f"  ⚠️ {warning}")
    
    # 4. 场景优化
    logger.info("可用的优化场景:")
    scenarios = ['conservative', 'aggressive', 'high_liquidity', 'small_cap']
    
    for scenario in scenarios:
        optimized_config = bt_config.optimize_for_scenario(scenario)
        logger.info(f"  {scenario}: 调仓{optimized_config.rebalancing_freq}, "
                   f"持仓{optimized_config.max_positions}只, "
                   f"最长{optimized_config.max_holding_days}天")
    
    # 5. 推荐最佳配置
    logger.info("=" * 60)
    logger.info("🎯 迁移建议:")
    logger.info("1. 对于现有策略，直接使用迁移后的配置")
    logger.info("2. 如果遇到Size小于100问题，Backtrader已自动解决")
    logger.info("3. 如果需要更好的停牌处理，启用重试机制")
    logger.info("4. 对于生产环境，推荐使用institutional_grade模板")
    logger.info("=" * 60)
    
    return bt_config


if __name__ == "__main__":
    # 演示配置迁移过程
    demo_config = demo_migration_process()
    
    # 显示所有可用模板
    templates = StrategyTemplates.get_all_templates()
    
    logger.info("所有可用模板:")
    for name, template in templates.items():
        logger.info(f"  {name}: {template.rebalancing_freq}调仓, {template.max_positions}只持仓")