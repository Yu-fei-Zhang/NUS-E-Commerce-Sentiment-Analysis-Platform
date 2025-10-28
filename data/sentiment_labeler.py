"""
基于情感词典的自动CRF标注工具
使用常见的正面和负面情感词来生成字符级标注
"""

import pandas as pd
import jieba
import json
from typing import List, Dict, Set

# 常见正面情感词词典
POSITIVE_WORDS = {
    # 直接正面词
    '好', '很好', '不错', '优秀', '精彩', '出色', '杰出', '卓越', '完美', '优质',
    '满意', '喜欢', '喜爱', '热爱', '推荐', '赞', '棒', '强', '牛', '厉害',
    '漂亮', '美丽', '精美', '华丽', '雅致', '精致', '细腻', '精良',
    '快', '快速', '迅速', '及时', '准时', '高效', '便捷', '方便',
    '清晰', '清楚', '明了', '明白', '明确', '透彻', '简洁', '简单',
    '实用', '有用', '好用', '易用', '适用', '管用', '顶用',
    '舒适', '舒服', '安逸', '惬意', '温馨', '和谐', '宁静',
    '值得', '超值', '划算', '实惠', '便宜', '经济', '合算',
    '新颖', '新奇', '创新', '独特', '特别', '有趣', '生动', '活泼',
    '丰富', '充实', '完整', '全面', '详细', '细致', '周到', '贴心',
    '高兴', '开心', '快乐', '愉快', '欢乐', '满足', '幸福', '欣慰',
    '感动', '震撼', '惊艳', '惊喜', '欣赏', '佩服', '崇拜',
    '成功', '顺利', '流畅', '稳定', '可靠', '靠谱', '放心',
    '专业', '权威', '正规', '正品', '真实', '可信', '诚信',
    '热情', '耐心', '周到', '友好', '亲切', '温和', '客气',
    '强大', '强悍', '给力', '威武', '霸气', '震撼', '惊人',
    '值', '赞美', '称赞', '表扬', '肯定', '认可', '支持',
    '好评', '五星', '满分', '优秀', '一流', '顶级', '高档',
    '惊艳', '惊叹', '惊讶', '赞叹', '称奇', '叫绝',
    '真好', '真棒', '真不错', '真的好', '非常好', '相当好', '特别好',
    '很棒', '很强', '很赞', '超棒', '超赞', '超好', '巨好',
    '太好了', '好极了', '棒极了', '爽', '赞爆了',
    '质量好', '品质好', '服务好', '态度好', '效果好',
    '有意思', '有内涵', '有深度', '有价值', '有意义',
    '准确', '精准', '到位', '合适', '恰当', '得体',
    '整洁', '干净', '卫生', '清爽', '利索',
    '厚实', '扎实', '牢固', '结实', '耐用', '持久',
    '轻便', '轻巧', '小巧', '灵活', '灵巧',
}

# 常见负面情感词词典
NEGATIVE_WORDS = {
    # 直接负面词
    '差', '很差', '不好', '糟糕', '恶劣', '低劣', '劣质', '次品',
    '失望', '遗憾', '可惜', '后悔', '郁闷', '无语', '无奈',
    '难', '困难', '麻烦', '复杂', '繁琐', '费劲', '吃力',
    '慢', '缓慢', '迟缓', '拖沓', '磨蹭', '延迟', '滞后',
    '贵', '昂贵', '太贵', '偏贵', '死贵', '坑', '宰人',
    '脏', '肮脏', '邋遢', '凌乱', '杂乱', '混乱',
    '旧', '陈旧', '老旧', '过时', '落后', '老套',
    '小', '狭小', '窄', '挤', '拥挤', '局促',
    '吵', '嘈杂', '喧闹', '吵闹', '噪音', '刺耳',
    '臭', '难闻', '恶心', '反胃', '呕吐', '作呕',
    '冷', '冰冷', '冷淡', '冷漠', '生硬', '僵硬',
    '热', '炎热', '闷热', '燥热', '烫', '烤',
    '硬', '坚硬', '生硬', '僵硬', '死板', '呆板',
    '软', '松软', '疲软', '绵软', '无力', '软趴趴',
    '薄', '单薄', '轻薄', '浅薄', '肤浅',
    '厚', '厚重', '笨重', '沉重', '累赘',
    '假', '虚假', '伪造', '仿冒', '盗版', '假货',
    '骗', '欺骗', '欺诈', '忽悠', '坑', '骗钱',
    '烂', '破烂', '腐烂', '糟烂', '垃圾', '渣',
    '坏', '损坏', '破坏', '毁坏', '弄坏', '坏了',
    '错', '错误', '失误', '过错', '不对', '有问题',
    '少', '缺少', '缺乏', '不足', '欠缺', '短缺',
    '漏', '遗漏', '漏掉', '缺漏', '疏漏',
    '掉', '掉色', '褪色', '脱落', '剥落',
    '裂', '破裂', '裂开', '开裂', '龟裂',
    '断', '折断', '弄断', '断裂', '中断',
    '松', '松动', '松散', '松懈', '松垮',
    '紧', '太紧', '紧绷', '紧巴', '勒',
    '粗糙', '简陋', '粗劣', '粗制滥造',
    '模糊', '不清', '不明', '混乱', '含糊',
    '卡', '卡顿', '卡死', '死机', '崩溃',
    '慢吞吞', '磨磨蹭蹭', '拖拖拉拉',
    '态度差', '服务差', '质量差', '效果差',
    '不满意', '不推荐', '不值', '不划算', '不实用',
    '太差了', '差劲', '很烂', '超烂', '巨烂',
    '难用', '不好用', '没用', '无用', '没意思',
    '无聊', '乏味', '枯燥', '单调', '平淡',
    '吵死了', '烦死了', '气死了', '郁闷死了',
    '受不了', '忍不了', '看不下去', '听不下去',
}

# 常见维度词词典（产品属性、特征等）
ASPECT_WORDS = {
    # 产品相关
    '质量', '品质', '做工', '工艺', '材质', '面料', '布料', '皮质',
    '外观', '外形', '造型', '款式', '设计', '样式', '风格', '颜色', '色彩',
    '尺寸', '大小', '尺码', '规格', '体积', '重量', '厚度', '长度', '宽度', '高度',
    '性能', '功能', '效果', '作用', '用途', '实用性', '耐用性',
    '价格', '价位', '定价', '售价', '成本', '性价比', '值不值',

    # 服务相关
    '服务', '态度', '服务态度', '客服', '售后', '售后服务',
    '物流', '快递', '配送', '送货', '速度', '时效',
    '包装', '打包', '包裹', '外包装', '内包装',

    # 书籍相关
    '内容', '故事', '情节', '剧情', '结局', '结尾',
    '文笔', '写作', '文字', '语言', '表达', '叙述',
    '翻译', '译文', '译者', '翻译质量',
    '印刷', '纸张', '纸质', '排版', '装帧', '插图',
    '作者', '作家', '主角', '人物', '角色',

    # 电子产品相关
    '屏幕', '显示', '画质', '画面', '色彩', '亮度',
    '音质', '音效', '声音', '音量', '喇叭', '扬声器',
    '电池', '续航', '电量', '待机', '充电',
    '系统', '软件', '界面', '操作', '反应', '流畅度',
    '配置', '硬件', '处理器', 'CPU', '内存', '存储',
    '相机', '摄像头', '拍照', '像素', '镜头',
    '信号', '网络', 'WiFi', '蓝牙', '连接',

    # 食品相关
    '味道', '口味', '口感', '口感', '风味', '滋味',
    '新鲜', '新鲜度', '保质期', '日期', '生产日期',
    '分量', '份量', '数量', '重量', '净重',
    '包装', '密封', '卫生', '干净',

    # 衣物相关
    '面料', '布料', '材质', '质地', '手感',
    '版型', '剪裁', '做工', '走线', '针脚',
    '尺码', '大小', '尺寸', '长短', '肥瘦',
    '颜色', '色差', '染色', '褪色', '掉色',

    # 住宿相关
    '环境', '氛围', '装修', '装饰', '设施',
    '位置', '地理位置', '交通', '地段',
    '房间', '空间', '面积', '布局',
    '卫生', '干净', '整洁', '清洁',
    '隔音', '噪音', '安静', '吵闹',

    # 其他通用
    '体验', '感觉', '感受', '印象',
}


class SentimentLabeler:
    """基于情感词典的CRF标注器"""

    def __init__(self):
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
        self.aspect_words = ASPECT_WORDS

        # 加载jieba用户词典
        for word in self.positive_words:
            jieba.add_word(word)
        for word in self.negative_words:
            jieba.add_word(word)
        for word in self.aspect_words:
            jieba.add_word(word)

    def find_word_in_text(self, text: str, word: str) -> List[int]:
        """
        在文本中查找词的起始位置（字符级别）
        返回所有匹配位置的列表
        """
        positions = []
        start = 0
        while True:
            pos = text.find(word, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    def label_text(self, text: str, sentiment_label: int) -> List[str]:
        """
        对文本进行字符级CRF标注

        Args:
            text: 输入文本
            sentiment_label: 情感标签 (1=负面, 2=正面, 3=中性)

        Returns:
            字符级标注列表
        """
        # 初始化标签为O
        labels = ['O'] * len(text)

        # 优先标注维度词（Aspect）
        for aspect_word in self.aspect_words:
            positions = self.find_word_in_text(text, aspect_word)
            for pos in positions:
                word_len = len(aspect_word)
                # 检查是否已被标注
                if labels[pos] == 'O':
                    labels[pos] = 'B-ASP'
                    for i in range(1, word_len):
                        if pos + i < len(labels) and labels[pos + i] == 'O':
                            labels[pos + i] = 'I-ASP'

        # 根据sentiment_label标注情感词
        if sentiment_label == 2:  # 正面
            sentiment_words = self.positive_words
            sent_tag = 'SENT_POS'
        elif sentiment_label == 1:  # 负面
            sentiment_words = self.negative_words
            sent_tag = 'SENT_NEG'
        else:  # 中性或其他
            # 尝试两种情感词
            sentiment_words = self.positive_words | self.negative_words
            sent_tag = None

        # 标注情感词
        for sent_word in sentiment_words:
            positions = self.find_word_in_text(text, sent_word)
            for pos in positions:
                word_len = len(sent_word)
                # 只标注未被标注的位置，避免覆盖维度词
                if labels[pos] == 'O':
                    # 确定情感极性
                    if sent_tag:
                        tag = sent_tag
                    else:
                        # 中性情况下，判断是正面还是负面词
                        if sent_word in self.positive_words:
                            tag = 'SENT_POS'
                        else:
                            tag = 'SENT_NEG'

                    labels[pos] = f'B-{tag}'
                    for i in range(1, word_len):
                        if pos + i < len(labels) and labels[pos + i] == 'O':
                            labels[pos + i] = f'I-{tag}'

        return labels

    def process_dataset(self, input_csv: str, output_csv: str, max_rows: int = None):
        """
        处理整个数据集

        Args:
            input_csv: 输入CSV文件路径
            output_csv: 输出CSV文件路径
            max_rows: 最大处理行数
        """
        print(f"正在读取文件: {input_csv}")

        # 读取CSV
        if max_rows:
            df = pd.read_csv(input_csv, nrows=max_rows)
        else:
            df = pd.read_csv(input_csv)

        print(f"共 {len(df)} 条数据待处理")

        # 处理每一行
        new_char_labels = []

        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"处理进度: {idx}/{len(df)}")

            content = row['content']
            sentiment_label = row['sentiment_label']

            # 处理空值
            if pd.isna(content) or not content:
                labels = []
            else:
                content = str(content)  # 确保是字符串
                # 生成新的字符级标注
                labels = self.label_text(content, sentiment_label)

            # 转换为JSON格式
            labels_json = json.dumps(labels, ensure_ascii=False)
            new_char_labels.append(labels_json)

        # 更新char_labels列
        df['char_labels'] = new_char_labels

        # 保存结果
        print(f"\n正在保存到: {output_csv}")
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print(f"✓ 标注完成！已保存 {len(df)} 条数据")

        # 统计标签分布
        self.print_statistics(new_char_labels)

    def print_statistics(self, all_char_labels: List[str]):
        """打印标签统计信息"""
        from collections import Counter

        all_labels = []
        for labels_json in all_char_labels:
            labels = json.loads(labels_json)
            all_labels.extend(labels)

        label_counts = Counter(all_labels)

        print("\n标签分布统计:")
        total = sum(label_counts.values())
        for label, count in sorted(label_counts.items()):
            percentage = (count / total) * 100
            print(f"  {label}: {count:,} ({percentage:.2f}%)")


def main():
    """主函数"""
    print("=" * 70)
    print("基于情感词典的CRF自动标注系统")
    print("=" * 70)

    # 初始化标注器
    labeler = SentimentLabeler()

    # 输入输出文件路径
    input_file = 'dataset.csv'
    output_file = 'dataset_relabeled.csv'

    # 处理数据集（可以先处理部分数据测试）
    print("\n开始处理数据集...")
    labeler.process_dataset(input_file, output_file)

    print("\n" + "=" * 70)
    print("处理完成！")
    print(f"输出文件: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
