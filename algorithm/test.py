"""
测试脚本
用于测试训练好的模型，包括鲁棒性测试和边界测试
"""
import torch
from transformers import BertTokenizerFast
from model import BertCRFModel
from config import Config
from inference import predict, format_results


def test_positive_samples():
    """测试正向评论识别"""
    print("\n" + "=" * 80)
    print("正向评论识别测试")
    print("=" * 80)

    positive_samples = [
        "这本书内容非常好，很有深度",
        "手机屏幕清晰，性能强大",
        "服务态度很好，环境舒适",
        "质量上乘，价格合理",
        "物流速度快，包装精美",
        "味道鲜美，分量十足",
        "操作简单，功能齐全",
        "设计精美，做工考究"
    ]

    return positive_samples


def test_negative_samples():
    """测试负向评论识别"""
    print("\n" + "=" * 80)
    print("负向评论识别测试")
    print("=" * 80)

    negative_samples = [
        "这本书内容很差，没有价值",
        "手机屏幕模糊，性能很烂",
        "服务态度恶劣，环境脏乱",
        "质量低劣，价格虚高",
        "物流速度慢，包装破损",
        "味道难吃，分量很少",
        "操作复杂，功能缺失",
        "设计丑陋，做工粗糙"
    ]

    return negative_samples


def test_mixed_samples():
    """测试混合情感识别"""
    print("\n" + "=" * 80)
    print("混合情感识别测试")
    print("=" * 80)

    mixed_samples = [
        "这家店的衣服面料差，但版型很好",
        "手机屏幕很清晰，但电池续航不行",
        "酒店服务态度很好，但环境一般",
        "这本书内容很充实，但是价格有点贵",
        "餐厅的菜品味道一般，但是环境不错",
        "质量还行，但价格太贵了",
        "外观设计漂亮，但功能很少",
        "物流很快，但包装简陋"
    ]

    return mixed_samples


def test_robustness():
    """鲁棒性测试：包含噪声、错别字等"""
    print("\n" + "=" * 80)
    print("鲁棒性测试")
    print("=" * 80)

    robustness_samples = [
        "这个手机真的太棒了！！！",  # 重复标点
        "服务态度非常非常非常好",  # 重复词语
        "味道真是好极了~~~",  # 特殊符号
        "質量很好，價格合理",  # 繁体字
        "屏幕清晰度真的没话说",  # 口语化表达
        "买的很值得，强烈推荐👍",  # 表情符号
        "性能杠杠的，完全满足需求",  # 网络用语
        "东西收到了，还不错哦~"  # 口语化
    ]

    return robustness_samples


def test_boundary_cases():
    """边界情况测试"""
    print("\n" + "=" * 80)
    print("边界情况测试")
    print("=" * 80)

    boundary_samples = [
        "还行",  # 极短文本
        "不错",  # 极短文本
        "一般般",  # 模糊评价
        "马马虎虎",  # 模糊评价
        "无功无过",  # 中性评价
        "质量",  # 单个属性词
        "好",  # 单个情感词
        "这个这个这个真的很好",  # 重复词
        "我觉得可能大概也许应该还不错吧",  # 不确定表达
    ]

    return boundary_samples


def test_long_text():
    """测试长文本"""
    print("\n" + "=" * 80)
    print("长文本测试")
    print("=" * 80)

    long_samples = [
        "这本书的内容非常丰富，作者的写作风格很独特，但是价格确实有点贵。不过考虑到书籍的质量和装帧，还是物有所值的。总的来说，我觉得这是一本值得推荐的好书，尽管有些章节略显冗长。",
        "手机的屏幕显示效果非常清晰，色彩还原度很高，看视频体验很好。电池续航能力也不错，正常使用一天完全没问题。但是摄像头的夜拍效果一般，在光线不好的情况下噪点比较明显。整体来说性价比还是挺高的。"
    ]

    return long_samples


def run_all_tests(model, tokenizer, config):
    """运行所有测试"""

    test_cases = {
        "正向评论": test_positive_samples(),
        "负向评论": test_negative_samples(),
        "混合情感": test_mixed_samples(),
        "鲁棒性": test_robustness(),
        "边界情况": test_boundary_cases(),
        "长文本": test_long_text()
    }

    for test_name, samples in test_cases.items():
        print("\n" + "=" * 80)
        print(f"{test_name}测试")
        print("=" * 80)

        for i, text in enumerate(samples, 1):
            print(f"\n测试样例 {i}:")
            results = predict(text, model, tokenizer, config)
            print(format_results(text, results))

            # 分析结果
            analyze_prediction(text, results, test_name)


def analyze_prediction(text: str, results: list, test_type: str):
    """分析预测结果"""
    if not results:
        print("⚠️  警告: 未识别到任何结果")
        return

    has_aspect = any(r['aspect_phrase'] for r in results)
    has_sentiment = any(r['sentiment_phrase'] for r in results)
    has_positive = any(r['sentiment'] == '正向' for r in results)
    has_negative = any(r['sentiment'] == '负向' for r in results)

    print("\n分析:")
    print(f"  - 识别到属性: {'✓' if has_aspect else '✗'}")
    print(f"  - 识别到情感: {'✓' if has_sentiment else '✗'}")

    if has_sentiment:
        if has_positive:
            print(f"  - 正向情感: ✓")
        if has_negative:
            print(f"  - 负向情感: ✓")

    # 根据测试类型进行特定分析
    if test_type == "正向评论" and not has_positive:
        print("  ⚠️  警告: 正向评论未识别到正向情感!")
    elif test_type == "负向评论" and not has_negative:
        print("  ⚠️  警告: 负向评论未识别到负向情感!")
    elif test_type == "混合情感" and not (has_positive and has_negative):
        print("  ⚠️  提示: 混合情感可能未完全识别")


def main():
    """主测试函数"""
    # 加载配置
    config = Config()
    print(config)

    # 加载tokenizer
    print("\n加载Tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)

    # 加载模型
    print("\n加载模型...")
    model_path = f"{config.output_dir}/best_model"

    try:
        model = BertCRFModel.load_pretrained(model_path, device=config.device)
        print("✓ 模型加载成功!")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("请先运行 train.py 训练模型")
        return

    # 运行所有测试
    run_all_tests(model, tokenizer, config)

    print("\n" + "=" * 80)
    print("所有测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()