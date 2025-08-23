"""
并发因子测试系统 - 安全且高效的多因子并行测试

核心特性:
1. 线程安全的数据隔离
2. 进程池并发，避免数据错乱
3. 实时进度监控
4. 失败重试机制
5. 结果汇总与报告

设计原理:
- 每个进程独立创建数据管理器实例，避免共享状态
- 使用进程池而非线程池，确保内存隔离
- 结果通过序列化传递，避免对象共享
"""

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import pickle
import os

from projects._03_factor_selection.config.config_file.load_config_file import _load_file
from quant_lib.config.logger_config import setup_logger

# 为每个进程设置日志
logger = setup_logger(__name__)


def t_single_factor_isolated(
    factor_name: str, 
    stock_pool_name: str, 
    config_path: str, 
    experiments_path: str,
    process_id: int,
    shared_progress: Optional[Any] = None
) -> Dict:
    """
    在独立进程中测试单个因子
    
    Args:
        factor_name: 因子名称
        stock_pool_name: 股票池名称  
        config_path: 配置文件路径
        experiments_path: 实验配置路径
        process_id: 进程ID
        shared_progress: 共享进度对象
        
    Returns:
        Dict: 测试结果字典
    """
    
    # 每个进程设置独立的日志
    process_logger = setup_logger(f"{__name__}_process_{process_id}")
    
    start_time = time.time()
    result = {
        'factor_name': factor_name,
        'stock_pool_name': stock_pool_name,
        'process_id': process_id,
        'status': 'failed',
        'error_message': None,
        'execution_time': 0,
        'test_results': None
    }
    
    try:
        process_logger.info(f"🚀 进程{process_id}: 开始测试因子 {factor_name}")
        
        # 在进程内部导入，避免跨进程共享
        from projects._03_factor_selection.data_manager.data_manager import DataManager
        from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
        from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer
        
        # 为每个进程创建独立的数据管理器实例
        process_logger.info(f"  📊 进程{process_id}: 初始化数据管理器...")
        data_manager = DataManager(
            config_path=config_path, 
            experiments_config_path=experiments_path
        )
        
        # 准备数据（每个进程独立加载）
        process_logger.info(f"  📈 进程{process_id}: 准备基础数据...")
        data_manager.prepare_basic_data()
        
        # 创建因子管理器
        factor_manager = FactorManager(data_manager)
        factor_manager.clear_cache()  # 确保缓存清洁
        
        # 创建因子分析器
        factor_analyzer = FactorAnalyzer(factor_manager=factor_manager)
        
        # 执行因子测试
        process_logger.info(f"  🔬 进程{process_id}: 执行因子测试...")
        test_results = factor_analyzer.test_factor_entity_service_route(
            factor_name=factor_name,
            stock_pool_index_name=stock_pool_name
        )
        
        # 提取关键结果（只保留可序列化的部分）
        serializable_results = extract_serializable_results(test_results, process_logger)
        
        result.update({
            'status': 'success',
            'test_results': serializable_results,
            'execution_time': time.time() - start_time
        })
        
        process_logger.info(f"✅ 进程{process_id}: 因子 {factor_name} 测试成功 "
                          f"(耗时: {result['execution_time']:.1f}秒)")
        
        # 更新共享进度
        if shared_progress:
            try:
                with shared_progress.get_lock():
                    shared_progress.value += 1
            except AttributeError:
                # 如果是ValueProxy对象，直接访问value
                shared_progress.value += 1
                
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        result.update({
            'status': 'failed', 
            'error_message': error_msg,
            'error_traceback': error_traceback,
            'execution_time': time.time() - start_time
        })
        
        process_logger.error(f"❌ 进程{process_id}: 因子 {factor_name} 测试失败: {error_msg}")
        process_logger.debug(f"详细错误信息:\\n{error_traceback}")
        
        # 更新共享进度
        if shared_progress:
            try:
                with shared_progress.get_lock():
                    shared_progress.value += 1
            except AttributeError:
                # 如果是ValueProxy对象，直接访问value
                shared_progress.value += 1
    
    return result


def extract_serializable_results(test_results: Dict, logger) -> Dict:
    """
    提取可序列化的测试结果
    避免传递复杂对象导致的序列化问题
    """
    
    serializable = {}
    
    try:
        # 提取IC相关结果
        if 'ic_stats' in test_results:
            serializable['ic_stats'] = test_results['ic_stats']
        
        # 提取分层回测结果
        if 'quantile_stats' in test_results:
            serializable['quantile_stats'] = test_results['quantile_stats']
        
        # 提取Fama-MacBeth结果
        if 'fm_stats' in test_results:
            serializable['fm_stats'] = test_results['fm_stats']
            
        # 提取综合得分
        if 'comprehensive_scores' in test_results:
            serializable['comprehensive_scores'] = test_results['comprehensive_scores']
            
        # 提取因子基础统计
        if 'factor_basic_stats' in test_results:
            serializable['factor_basic_stats'] = test_results['factor_basic_stats']
            
    except Exception as e:
        logger.warning(f"提取序列化结果时出错: {e}")
        serializable['extraction_error'] = str(e)
    
    return serializable


class ConcurrentFactorTester:
    """并发因子测试器"""
    
    def __init__(self, config_path: str, experiments_path: str, max_workers: int = 4):
        """
        初始化并发测试器
        
        Args:
            config_path: 配置文件路径
            experiments_path: 实验配置路径  
            max_workers: 最大并发进程数
        """
        self.config_path = str(Path(config_path).absolute())
        self.experiments_path = str(Path(experiments_path).absolute())
        self.max_workers = max_workers
        self.results = []
        
        logger.info(f"🔧 初始化并发因子测试器 (最大并发数: {max_workers})")
        
    def run_batch_testing(self, test_configs: List[Dict]) -> List[Dict]:
        """
        执行批量并发测试
        
        Args:
            test_configs: 测试配置列表，每个包含 factor_name 和 stock_pool_name
            
        Returns:
            List[Dict]: 所有测试结果
        """
        
        total_tests = len(test_configs)
        logger.info(f"🚀 开始并发批量测试 {total_tests} 个因子...")
        
        # 创建共享进度计数器
        with Manager() as manager:
            shared_progress = manager.Value('i', 0)
            
            # 创建进程池并提交任务
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                
                # 提交所有测试任务
                future_to_config = {}
                for i, config in enumerate(test_configs):
                    future = executor.submit(
                        t_single_factor_isolated,
                        factor_name=config['factor_name'],
                        stock_pool_name=config['stock_pool_name'],
                        config_path=self.config_path,
                        experiments_path=self.experiments_path,
                        process_id=i,
                        shared_progress=shared_progress
                    )
                    future_to_config[future] = config
                
                # 监控执行进度
                results = []
                start_time = time.time()
                
                for future in as_completed(future_to_config):
                    result = future.result()
                    results.append(result)
                    
                    # 实时进度报告
                    completed = len(results)
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    eta = avg_time * (total_tests - completed)
                    
                    if result['status'] == 'success':
                        logger.info(f"✅ [{completed}/{total_tests}] {result['factor_name']} 完成 "
                                  f"(耗时: {result['execution_time']:.1f}s, 预计剩余: {eta:.1f}s)")
                    else:
                        logger.error(f"❌ [{completed}/{total_tests}] {result['factor_name']} 失败: "
                                   f"{result['error_message']}")
        
        self.results = results
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: List[Dict]):
        """生成测试汇总报告"""
        
        total_tests = len(results)
        successful_tests = [r for r in results if r['status'] == 'success']
        failed_tests = [r for r in results if r['status'] == 'failed']
        
        total_time = sum(r['execution_time'] for r in results)
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        logger.info("\\n" + "="*60)
        logger.info("📊 并发因子测试汇总报告")
        logger.info("="*60)
        logger.info(f"🎯 总测试数量: {total_tests}")
        logger.info(f"✅ 成功测试: {len(successful_tests)} ({len(successful_tests)/total_tests*100:.1f}%)")
        logger.info(f"❌ 失败测试: {len(failed_tests)} ({len(failed_tests)/total_tests*100:.1f}%)")
        logger.info(f"⏱️  总执行时间: {total_time:.1f}秒")
        logger.info(f"📈 平均测试时间: {avg_time:.1f}秒")
        
        if failed_tests:
            logger.info(f"\\n❌ 失败因子列表:")
            for test in failed_tests:
                logger.info(f"  - {test['factor_name']}: {test['error_message']}")
        
        if successful_tests:
            logger.info(f"\\n🏆 成功因子前5名 (按执行时间):")
            sorted_successful = sorted(successful_tests, key=lambda x: x['execution_time'])
            for i, test in enumerate(sorted_successful[:5], 1):
                logger.info(f"  {i}. {test['factor_name']}: {test['execution_time']:.1f}秒")
    
    def save_results(self, output_path: Optional[str] = None):
        """保存测试结果"""
        
        if not self.results:
            logger.warning("没有可保存的测试结果")
            return
        
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"concurrent_factor_test_results_{timestamp}.json"
        
        try:
            # 准备保存的数据
            save_data = {
                'metadata': {
                    'test_timestamp': pd.Timestamp.now().isoformat(),
                    'total_factors': len(self.results),
                    'successful_factors': len([r for r in self.results if r['status'] == 'success']),
                    'config_path': self.config_path,
                    'max_workers': self.max_workers
                },
                'results': self.results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 测试结果已保存至: {output_path}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")


def create_test_configs_from_experiments(experiments_path: str) -> List[Dict]:
    """从实验配置文件创建测试配置"""
    
    # 这里需要根据你的experiments.yaml格式来调整
    try:
        import yaml
        with open(experiments_path, 'r', encoding='utf-8') as f:
            experiments = yaml.safe_load(f)
        
        test_configs = []
        
        # 根据你的实验配置格式提取因子和股票池配置
        if 'experiments' in experiments:
            for exp in experiments['experiments']:
                test_configs.append({
                    'factor_name': exp['factor_name'],
                    'stock_pool_name': exp.get('stock_pool_name', 'institutional_stock_pool')
                })
        
        return test_configs
        
    except Exception as e:
        logger.error(f"解析实验配置失败: {e}")
        return []


def main():
    """并发测试主函数"""
    
    try:
        # 配置文件路径
        current_dir = Path(__file__).parent
        config_path = 'D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\config.yaml'
        experiments_path = 'D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\experiments.yaml'

        test_configs = _load_file('D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\experiments.yaml')

        # 创建并发测试器
        # 建议并发数不要超过CPU核心数的75%
        import multiprocessing
        max_workers = min(3, max(1, multiprocessing.cpu_count() // 4))
        
        logger.info(f"🔧 系统CPU核心数: {multiprocessing.cpu_count()}, 使用并发数: {max_workers}")
        
        tester = ConcurrentFactorTester(
            config_path=str(config_path),
            experiments_path=str(experiments_path), 
            max_workers=max_workers
        )
        
        # 执行并发测试
        logger.info(f"🚀 开始并发测试 {len(test_configs)} 个因子...")
        start_time = time.time()
        
        results = tester.run_batch_testing(test_configs)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 所有测试完成! 总耗时: {total_time:.1f}秒")
        
        # 保存结果
        tester.save_results()
        
        return results
        
    except Exception as e:
        logger.error(f"并发测试执行失败: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    results = main()