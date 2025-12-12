"""
修复txt文件中的Windows路径，转换为Linux路径
将反斜杠替换为正斜杠，并更新路径前缀
"""
import os
import re

def fix_paths_in_file(file_path, old_base_path, new_base_path):
    """
    修复文件中的路径
    
    Args:
        file_path: 要修复的txt文件路径
        old_base_path: 旧的Windows路径前缀（用于匹配和替换）
        new_base_path: 新的Linux路径前缀
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，跳过")
        return
    
    print(f"正在处理: {file_path}")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行
    fixed_lines = []
    for line_num, line in enumerate(lines, 1):
        original_line = line.strip()
        if not original_line:
            fixed_lines.append(line)
            continue
        
        # 替换所有反斜杠为正斜杠（包括单个和双个）
        fixed_line = line.replace('\\\\', '/').replace('\\', '/')
        
        # 替换Windows路径前缀为Linux路径前缀
        # 匹配 D:/ 或 D:// 开头的路径
        fixed_line = re.sub(r'D:[/\\]+', new_base_path, fixed_line)
        
        # 如果还有旧的Windows路径格式，也替换
        fixed_line = re.sub(
            r'D:\\+1JinKuang\\+Segmentation\\+Segmentation\\+datasets',
            new_base_path + '/datasets',
            fixed_line
        )
        
        # 确保路径格式正确（移除多余的正斜杠）
        fixed_line = re.sub(r'/+', '/', fixed_line)
        
        fixed_lines.append(fixed_line + '\n')
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"  ✓ 完成，共处理 {len(lines)} 行")

def main():
    # 配置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 旧路径前缀（Windows格式）
    old_windows_path = r'D:\\1JinKuang\\Segmentation\\Segmentation\\datasets'
    
    # 新路径前缀（Linux格式）
    new_linux_base = '/root/data1/paper/VisionLab'
    
    # 要处理的文件列表
    files_to_fix = [
        'kvasir_seg_train.txt',
        'kvasir_seg_val.txt'
    ]
    
    print("=" * 60)
    print("开始修复txt文件中的路径")
    print("=" * 60)
    print(f"旧路径前缀: {old_windows_path}")
    print(f"新路径前缀: {new_linux_base}/datasets")
    print("=" * 60)
    
    # 处理每个文件
    for filename in files_to_fix:
        file_path = os.path.join(base_dir, filename)
        fix_paths_in_file(file_path, old_windows_path, new_linux_base)
    
    print("=" * 60)
    print("所有文件处理完成！")
    print("=" * 60)
    
    # 验证：显示第一行作为示例
    print("\n验证结果（显示每个文件的第一行）:")
    for filename in files_to_fix:
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    print(f"\n{filename}:")
                    print(f"  {first_line}")

if __name__ == '__main__':
    main()

