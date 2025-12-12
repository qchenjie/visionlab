"""
训练脚本 - 执行三个训练任务
1. missformer 在 kvasir_seg 数据集上训练
2. swin_unet 在 Sypase 数据集上训练
3. setr 在 Sypase 数据集上训练
"""
import subprocess
import sys
import os

def run_training_task(task_name, cmd):
    """执行单个训练任务"""
    print("\n" + "=" * 60)
    print(f"开始训练任务: {task_name}")
    print("=" * 60)
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n任务 '{task_name}' 训练完成！")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n任务 '{task_name}' 训练过程中出现错误: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n任务 '{task_name}' 被用户中断")
        return 1

def main():
    tasks = []
    
    # 任务1: missformer 在 kvasir_seg 数据集上训练
    tasks.append({
        'name': 'missformer on kvasir_seg',
        'cmd': [
            sys.executable,
            'train.py',
            '--save', '100',
            '--freeze_epoch', '500',
            '--unfreeze_epoch', '2000',
            '--datasetname', 'kvasir_seg',
            '--bs', '32',
            '--train_txt', 'kvasir_seg_train.txt',
            '--test_txt', 'kvasir_seg_val.txt',
            '--val_txt', 'kvasir_seg_val.txt',
            '--method', 'missformer'
        ]
    })
    
    # 任务2: swin_unet 在 Sypase 数据集上训练
    tasks.append({
        'name': 'swin_unet on Sypase',
        'cmd': [
            sys.executable,
            'train.py',
            '--save', '10',
            '--freeze_epoch', '250',
            '--init_epoch', '0',
            '--unfreeze_epoch', '800',
            '--datasetname', 'Sypase',
            '--bs', '32',
            '--train_txt', 'train.txt',
            '--test_txt', 'test.txt',
            '--num_worker', '4',
            '--val_txt', 'test.txt',
            '--method', 'swin_unet'
        ]
    })
    
    # 任务3: setr 在 Sypase 数据集上训练
    tasks.append({
        'name': 'setr on Sypase',
        'cmd': [
            sys.executable,
            'train.py',
            '--save', '10',
            '--freeze_epoch', '250',
            '--init_epoch', '0',
            '--unfreeze_epoch', '800',
            '--datasetname', 'Sypase',
            '--bs', '32',
            '--train_txt', 'train.txt',
            '--test_txt', 'test.txt',
            '--num_worker', '4',
            '--val_txt', 'test.txt',
            '--method', 'setr'
        ]
    })
    
    # 按顺序执行所有任务
    print("=" * 60)
    print(f"准备执行 {len(tasks)} 个训练任务")
    print("=" * 60)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n>>> 任务 {i}/{len(tasks)}")
        exit_code = run_training_task(task['name'], task['cmd'])
        if exit_code != 0:
            print(f"\n任务 {i} 执行失败，退出码: {exit_code}")
            return exit_code
    
    print("\n" + "=" * 60)
    print("所有训练任务完成！")
    print("=" * 60)
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
