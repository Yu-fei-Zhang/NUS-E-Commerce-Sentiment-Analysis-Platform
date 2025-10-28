import torch
import os

def diagnose_model_file(model_path):
    """
    诊断模型文件问题
    """
    print(f"\n正在诊断模型文件: {model_path}")
    print("="*60)

    # 1. 检查文件是否存在
    if not os.path.exists(model_path):
        print("❌ 文件不存在")
        return False

    print(f"✓ 文件存在")

    # 2. 检查文件大小
    file_size = os.path.getsize(model_path)
    print(f"✓ 文件大小: {file_size / (1024**2):.2f} MB")

    if file_size < 1000:  # 小于1KB可能是空文件
        print("❌ 文件太小,可能是损坏的")
        return False

    # 3. 尝试以不同方式加载
    try:
        # 尝试加载
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✓ 模型加载成功")
        print(f"✓ 包含 {len(state_dict)} 个参数")
        return True
    except KeyError as e:
        print(f"❌ KeyError: {e}")
        print("   文件已损坏,需要重新训练")
        return False
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


# 使用诊断功能
diagnose_model_file('saved_model/best_model/best_model.pt')