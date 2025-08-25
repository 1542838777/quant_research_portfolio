"""
增强的测试运行器 - 集成配置快照管理

核心改进：
1. 每次测试后自动保存配置快照
2. 测试结果与配置快照自动关联
3. 提供配置回溯和对比功能
4. 测试历史的完整追踪

使用方式：
- 替代原有的example_usage.py
- 完全兼容原有测试流程
- 自动化配置管理，无需手动干预
"""

import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from projects._03_factor_selection.config.config_file.load_config_file import _load_local_config_functional
# 原有的导入
from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager

# 新增：配置快照管理器
from projects._03_factor_selection.factory.config_snapshot_manager import (
    ConfigSnapshotManager, 
    load_config_from_yaml
)

from quant_lib.config.logger_config import setup_logger, log_success
from quant_lib.utils.test import check_step

# 配置日志
logger = setup_logger(__name__)


class EnhancedTestRunner:
    """增强的测试运行器"""
    
    def __init__(self, config_path: str, experiments_config_path: str):
        """
        初始化测试运行器
        
        Args:
            config_path: 主配置文件路径
            experiments_config_path: 实验配置文件路径
        """
        self.config_path = Path(config_path)
        self.experiments_config_path = Path(experiments_config_path)
        self.workspace_root = self.config_path.parent.parent / "workspace"
        
        # 初始化配置快照管理器
        self.config_snapshot_manager = ConfigSnapshotManager(str(self.workspace_root))
        
        # 加载配置
        self.config = _load_local_config_functional(str(self.config_path))
        
        # 初始化核心组件
        self.data_manager = None
        self.factor_manager = None
        self.factor_analyzer = None
        
        # 当前测试会话信息
        self.current_session = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'test_count': 0,
            'snapshot_id': None
        }
    
    def initialize_managers(self):
        """初始化各种管理器"""
        logger.info("🔄 开始初始化测试环境...")
        
        # 初始化数据管理器
        logger.info("1. 初始化数据管理器...")
        self.data_manager = DataManager(
            config_path=str(self.config_path), 
            experiments_config_path=str(self.experiments_config_path)
        )
        self.data_manager.prepare_basic_data()
        
        # 初始化因子管理器
        logger.info("2. 初始化因子管理器...")
        self.factor_manager = FactorManager(self.data_manager)
        self.factor_manager.clear_cache()
        
        # 初始化因子分析器
        logger.info("3. 初始化因子分析器...")
        self.factor_analyzer = FactorAnalyzer(factor_manager=self.factor_manager)
        
        logger.info("✅ 测试环境初始化完成")
    
    def create_session_snapshot(self, session_description: str = ""):
        """为当前测试会话创建配置快照"""
        try:
            logger.info("📸 创建测试会话配置快照...")
            
            # 构建快照名称
            snapshot_name = f"测试会话_{self.current_session['session_id']}"
            if session_description:
                snapshot_name += f"_{session_description}"
            
            # 创建测试上下文
            experiments_df = self.data_manager.get_experiments_df()
            test_context = {
                'session_id': self.current_session['session_id'],
                'session_description': session_description,
                'total_experiments': len(experiments_df),
                'factor_names': experiments_df['factor_name'].unique().tolist(),
                'stock_pools': experiments_df['stock_pool_name'].unique().tolist(),
                'backtest_period': f"{self.config.get('backtest', {}).get('start_date')} - {self.config.get('backtest', {}).get('end_date')}",
                'created_by': 'EnhancedTestRunner',
                'runtime_modifications': self._detect_runtime_modifications()
            }
            
            # 创建快照
            snapshot_id = self.config_snapshot_manager.create_snapshot(
                config=self.config,
                snapshot_name=snapshot_name,
                test_context=test_context
            )
            
            self.current_session['snapshot_id'] = snapshot_id
            logger.info(f"✅ 配置快照创建完成: {snapshot_id}")
            
            return snapshot_id
            
        except Exception as e:
            logger.error(f"❌ 创建配置快照失败: {e}")
            return None
    
    def run_batch_tests(self, session_description: str = "批量因子测试") -> List[Dict]:
        """
        运行批量测试并自动管理配置快照
        
        Args:
            session_description: 测试会话描述
            
        Returns:
            List[Dict]: 测试结果列表
        """
        logger.info(f"🚀 开始批量测试会话: {session_description}")
        
        # 1. 初始化管理器
        self.initialize_managers()
        
        # 2. 创建会话配置快照
        session_snapshot_id = self.create_session_snapshot(session_description)
        if not session_snapshot_id:
            raise ValueError("⚠️ 配置快照创建失败，继续测试但无法追踪配置")
        
        # 3. 准备实验配置
        experiments_df = self.data_manager.get_experiments_df()
        logger.info(f"📊 准备执行 {len(experiments_df)} 个实验")
        
        # 4. 保存价格数据
        self._save_close_hfq_if_needed(experiments_df)
        
        # 5. 执行批量测试
        results = []
        successful_tests = 0
        
        for index, config in experiments_df.iterrows():
            try:
                factor_name = config['factor_name']
                stock_pool_name = config['stock_pool_name']
                
                logger.info(f"🧪 [{index+1}/{len(experiments_df)}] 测试因子: {factor_name} (股票池: {stock_pool_name})")
                
                # 执行单个因子测试
                test_result = self._run_single_factor_test(
                    factor_name=factor_name,
                    stock_pool_name=stock_pool_name,
                    session_snapshot_id=session_snapshot_id
                )
                
                results.append({
                    'factor_name': factor_name,
                    'stock_pool_name': stock_pool_name,
                    'result': test_result,
                    'snapshot_id': session_snapshot_id,
                    'test_timestamp': datetime.now().isoformat()
                })
                
                successful_tests += 1
                self.current_session['test_count'] += 1
                
                logger.info(f"✅ [{index+1}/{len(experiments_df)}] 因子 {factor_name} 测试完成")
                
            except Exception as e:
                logger.error(f"❌ [{index+1}/{len(experiments_df)}] 因子 {factor_name} 测试失败: {e}")
                results.append({
                    'factor_name': factor_name,
                    'stock_pool_name': stock_pool_name,
                    'result': None,
                    'error': str(e),
                    'snapshot_id': session_snapshot_id,
                    'test_timestamp': datetime.now().isoformat()
                })
                
                # 根据需要决定是否继续
                if not self._should_stop_on_error():
                    raise ValueError("⚠️ 遇到错误，停止批量测试")

        # 6. 生成测试会话摘要
        self._generate_session_summary(results, successful_tests, session_snapshot_id)
        
        log_success(f"✅ 批量测试会话完成: 成功 {successful_tests}/{len(experiments_df)} 个因子")
        return results
    
    def _run_single_factor_test(
        self, 
        factor_name: str, 
        stock_pool_name: str,
        session_snapshot_id: Optional[str]
    ) -> Any:
        """
        执行单个因子测试并关联配置快照
        
        Args:
            factor_name: 因子名称
            stock_pool_name: 股票池名称  
            session_snapshot_id: 会话配置快照ID
            
        Returns:
            测试结果
        """
        # 执行因子测试
        test_result = self.factor_analyzer.test_factor_entity_service_route(
            factor_name=factor_name,
            stock_pool_index_name=stock_pool_name,
        )
        
        # 关联配置快照
        if session_snapshot_id:
            try:
                stock_pool_index = self.factor_manager.data_manager.get_stock_pool_index_code_by_name(stock_pool_name)
                
                success = self.config_snapshot_manager.link_test_result(
                    snapshot_id=session_snapshot_id,
                    factor_name=factor_name,
                    stock_pool=stock_pool_index,
                    calc_type='c2c',  # 默认值，可以从配置中读取
                    version=f"{self.data_manager.backtest_start_date}_{self.data_manager.backtest_end_date}",
                    test_description=f"批量测试_{self.current_session['session_id']}"
                )
                
                if success:
                    logger.debug(f"✅ 配置快照关联成功: {factor_name} -> {session_snapshot_id}")
                else:
                    logger.warning(f"⚠️ 配置快照关联失败: {factor_name}")
                    
            except Exception as e:
                logger.error(f"❌ 配置快照关联异常: {factor_name} - {e}")
        
        return test_result
    
    def _save_close_hfq_if_needed(self, experiments_df: pd.DataFrame):
        try:
            # 获取第一个实验的股票池（用于保存价格数据）
            first_stock_pool = experiments_df.iloc[0]['stock_pool_name']#这句话 就限制了 第一个实验的股票池 ：即：当前设计 不不支持一次实验使用多个不同的股票池！
            stock_pool_index_code = self.factor_manager.data_manager.get_stock_pool_index_code_by_name(first_stock_pool)
            
            close_hfq = self.factor_manager.get_prepare_aligned_factor_for_analysis('close_hfq', first_stock_pool, True)
            if close_hfq is None:
                raise ValueError("close_hfq 数据为空，无法保存")
            
            path = Path(
                r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace\result"
            ) / stock_pool_index_code / 'close_hfq' / f'{self.data_manager.backtest_start_date}_{self.data_manager.backtest_end_date}'
            
            path.mkdir(parents=True, exist_ok=True)
            close_hfq.to_parquet(path / 'close_hfq.parquet')
            logger.info(f"📊 价格数据保存成功: {path / 'close_hfq.parquet'}")
            
        except Exception as e:
            logger.warning(f"⚠️ 价格数据保存失败: {e}")
    
    def _detect_runtime_modifications(self) -> Dict[str, Any]:
        """检测运行时的配置修改"""
        # 这里可以实现检测运行时配置修改的逻辑
        # 例如：对比原始配置文件和当前配置的差异
        modifications = {
            'detected_at': datetime.now().isoformat(),
            'modifications': [],
            'notes': '暂未实现运行时修改检测'
        }
        return modifications
    
    def _should_stop_on_error(self) -> bool:
        """判断是否在遇到错误时停止测试"""
        # 可以根据配置或者错误类型来决定
        return True
    
    def _generate_session_summary(
        self, 
        results: List[Dict], 
        successful_tests: int,
        snapshot_id: Optional[str]
    ):
        """生成测试会话摘要"""
        try:
            summary = {
                'session_id': self.current_session['session_id'],
                'snapshot_id': snapshot_id,
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(results),
                'successful_tests': successful_tests,
                'failed_tests': len(results) - successful_tests,
                'success_rate': successful_tests / len(results) if results else 0,
                'factors_tested': [r['factor_name'] for r in results],
                'stock_pools_used': list(set(r['stock_pool_name'] for r in results)),
                'test_duration': 'unknown',  # 可以添加计时逻辑
                'config_snapshot_id': snapshot_id
            }
            
            # 保存会话摘要
            summary_path = self.workspace_root / "test_sessions" / f"session_{self.current_session['session_id']}.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📋 测试会话摘要已保存: {summary_path}")
            
            # 打印摘要
            self._print_session_summary(summary)
            
        except Exception as e:
            logger.error(f"❌ 生成会话摘要失败: {e}")
    
    def _print_session_summary(self, summary: Dict):
        """打印测试会话摘要"""
        print(f"\n{'='*60}")
        print(f"📊 测试会话摘要")
        print(f"{'='*60}")
        print(f"🆔 会话ID: {summary['session_id']}")
        print(f"📸 配置快照: {summary['snapshot_id']}")
        print(f"⏰ 完成时间: {summary['timestamp']}")
        print(f"🧪 测试总数: {summary['total_tests']}")
        print(f"✅ 成功数量: {summary['successful_tests']}")
        print(f"❌ 失败数量: {summary['failed_tests']}")
        print(f"📈 成功率: {summary['success_rate']:.1%}")
        
        print(f"\n📋 测试的因子:")
        for factor in summary['factors_tested']:
            print(f"  • {factor}")
        
        print(f"\n📊 使用的股票池:")
        for pool in summary['stock_pools_used']:
            print(f"  • {pool}")
        
        print(f"{'='*60}")
    
    def get_test_history(self, limit: int = 10) -> List[Dict]:
        """获取测试历史"""
        try:
            sessions_dir = self.workspace_root / "test_sessions"
            if not sessions_dir.exists():
                return []
            
            sessions = []
            for session_file in sessions_dir.glob("session_*.json"):
                try:
                    import json
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        sessions.append(session_data)
                except Exception as e:
                    logger.warning(f"读取会话文件失败 {session_file}: {e}")
            
            # 按时间倒序排序
            sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return sessions[:limit]
            
        except Exception as e:
            logger.error(f"获取测试历史失败: {e}")
            return []
    
    def compare_test_configs(self, session_id1: str, session_id2: str) -> Dict:
        """比较两个测试会话的配置差异"""
        # 先获取两个会话的快照ID
        sessions = self.get_test_history(limit=50)
        
        snapshot_id1 = None
        snapshot_id2 = None
        
        for session in sessions:
            if session['session_id'] == session_id1:
                snapshot_id1 = session.get('snapshot_id')
            elif session['session_id'] == session_id2:
                snapshot_id2 = session.get('snapshot_id')
        
        if not snapshot_id1 or not snapshot_id2:
            return {'error': '找不到对应的配置快照'}
        
        # 比较配置差异
        return self.config_snapshot_manager.compare_configs(snapshot_id1, snapshot_id2)


def main():
    """主函数 - 使用增强的测试运行器"""
    try:
        # 配置路径
        current_dir = Path(__file__).parent
        config_path = str(current_dir / 'config.yaml') #todo check
        experiments_config_path = str(current_dir / 'experiments.yaml')
        
        # 创建增强的测试运行器
        test_runner = EnhancedTestRunner(config_path, experiments_config_path)
        
        # 运行批量测试
        results = test_runner.run_batch_tests(
            session_description="生产环境_因子筛选_V2.0"
        )
        
        # 输出最终结果
        logger.info(f"🎉 测试完成！共处理 {len(results)} 个因子")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 测试运行失败: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()