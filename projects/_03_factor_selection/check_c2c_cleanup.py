"""
检查C2C函数清理情况
确保所有calcu_forward_returns_close_close的引用都已被清理
"""

import os
import re
from pathlib import Path

def search_c2c_references(root_dir):
    """搜索所有C2C函数的引用"""
    
    c2c_references = []
    
    # 搜索模式
    patterns = [
        r'calcu_forward_returns_close_close',
        r'c2c_calculator',
        r"'c2c'",
        r'"c2c"'
    ]
    
    # 排除的目录
    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.vscode'}
    
    # 搜索文件
    for root, dirs, files in os.walk(root_dir):
        # 排除特定目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith(('.py', '.yaml', '.yml', '.txt', '.md')):
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for i, line in enumerate(content.split('\n'), 1):
                        for pattern in patterns:
                            if re.search(pattern, line) and not line.strip().startswith('#'):
                                c2c_references.append({
                                    'file': str(file_path),
                                    'line': i,
                                    'content': line.strip(),
                                    'pattern': pattern
                                })
                                
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
    
    return c2c_references

def check_config_files():
    """检查配置文件中的C2C引用"""
    
    config_files = [
        'projects/_03_factor_selection/factory/config.yaml',
        'projects/_03_factor_selection/factory/experiments.yaml'
    ]
    
    print("🔍 检查配置文件...")
    
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'c2c' in content.lower():
                print(f"⚠️  {config_file} 中仍包含c2c引用")
                for i, line in enumerate(content.split('\n'), 1):
                    if 'c2c' in line.lower():
                        print(f"   第{i}行: {line.strip()}")
            else:
                print(f"✅ {config_file} 已清理")
        else:
            print(f"❌ 配置文件不存在: {config_file}")

def main():
    """主函数"""
    
    print("🚀 C2C函数清理检查")
    print("=" * 60)
    
    # 获取项目根目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    print(f"搜索目录: {project_root}")
    print()
    
    # 搜索C2C引用
    print("🔍 搜索C2C函数引用...")
    references = search_c2c_references(project_root)
    
    if references:
        print(f"❌ 发现 {len(references)} 个C2C引用:")
        print()
        
        for ref in references:
            print(f"文件: {ref['file']}")
            print(f"行号: {ref['line']}")
            print(f"内容: {ref['content']}")
            print(f"匹配: {ref['pattern']}")
            print("-" * 40)
    else:
        print("✅ 未发现活跃的C2C引用")
    
    print()
    
    # 检查配置文件
    check_config_files()
    
    print()
    print("🎯 清理建议:")
    
    if references:
        print("1. 将所有calcu_forward_returns_close_close替换为calcu_forward_returns_open_close")
        print("2. 更新函数参数：添加open_df参数")
        print("3. 清理配置文件中的'c2c'引用")
        print("4. 注释或删除调试代码中的C2C调用")
    else:
        print("✅ C2C函数已完全清理")
        print("✅ 可以重新运行单调性测试")
        print("✅ 1日收益率异常问题应该已解决")
    
    print()
    print("🔄 下一步:")
    print("1. 重新运行单因子测试")
    print("2. 检查1日收益率的单调性是否正常")
    print("3. 验证volatility因子的表现")

if __name__ == "__main__":
    main()
