# ==================== 工具函数模块 ====================
import pandas as pd
from flask import current_app
from model_inference import analyze_text, analyze_batch as model_analyze_batch


def allowed_file(filename):
    """检查文件扩展名"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def read_file(filepath):
    """Read CSV or Excel file"""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, encoding='utf-8')
        else:
            df = pd.read_excel(filepath)

        # Get comment content from first column
        comments = df.iloc[:, 0].dropna().astype(str).tolist()
        return comments[:100]  # Limit to 100
    except Exception as e:
        print(f"File reading error: {e}")
        return []


def analyze_sentiment(text):
    """
    Use real model for sentiment analysis
    Return format: {'sentiment': 'Positive/Negative/Neutral', 'entities': [...], 'aspect_sentiments': [...]}
    """
    try:
        result = analyze_text(text)

        # Convert sentiment labels to Chinese
        sentiment_map = {
            'positive': '正面',
            'negative': '负面',
            'neutral': '中性'
        }

        sentiment_cn = sentiment_map.get(result['sentiment'], '中性')

        # Convert aspect_sentiments to Chinese
        aspect_sentiments_cn = []
        for asp_sent in result.get('aspect_sentiments', []):
            aspect_sentiments_cn.append({
                'aspect': asp_sent['aspect'],
                'sentiment': sentiment_map.get(asp_sent['sentiment'], '中性'),
                'sentiment_words': asp_sent['sentiment_words']
            })

        return {
            'sentiment': sentiment_cn,
            'entities': result['entities'],
            'sentiment_counts': result['sentiment_counts'],
            'aspect_sentiments': aspect_sentiments_cn
        }
    except Exception as e:
        print(f"Model analysis error: {e}")
        print("Using rule-based fallback analysis")
        # Use rule-based analysis when model fails
        return rule_based_sentiment_analysis(text)


def rule_based_sentiment_analysis(text):
    """
    Simple rule-based sentiment analysis for demo
    When trained model is unavailable
    """
    # Positive keywords
    positive_words = [
        '好', '很好', '非常好', '优秀', '完美', '满意', '喜欢', '不错', '赞', '棒',
        '值得', '推荐', '舒服', '漂亮', '精致', '高端', '惊喜', '超值', '实惠',
        '快速', '及时', '专业', '贴心', '细心', '周到', '热情', '友好', '靠谱',
        '质量好', '做工精细', '包装完好', '物流快', '态度好', '服务好'
    ]

    # Negative keywords
    negative_words = [
        '差', '很差', '不好', '失望', '糟糕', '不满意', '难看', '难用', '坏',
        '烂', '退货', '投诉', '垃圾', '劣质', '假货', '骗人', '后悔', '气人',
        '破损', '瑕疵', '掉漆', '褪色', '缩水', '变形', '难闻', '刺鼻',
        '慢', '态度差', '不耐烦', '敷衍', '推诿', '不理', '忽悠', '欺骗',
        '质量差', '做工粗糙', '包装破损', '物流慢', '服务差', '不专业'
    ]

    # Calculate sentiment scores
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)

    # Determine sentiment
    if positive_count > negative_count:
        sentiment = '正面'
    elif negative_count > positive_count:
        sentiment = '负面'
    else:
        sentiment = '中性'

    return {
        'sentiment': sentiment,
        'entities': [],
        'sentiment_counts': {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': 0
        },
        'aspect_sentiments': []
    }


def analyze_batch(comments):
    """
    Batch analyze comments
    Return: (result list, statistics)
    """
    try:
        # Use model for batch analysis
        raw_results = model_analyze_batch(comments)

        # Convert to format needed by system
        results = []
        stats = {'positive': 0, 'negative': 0, 'neutral': 0}

        sentiment_map = {
            'positive': '正面',
            'negative': '负面',
            'neutral': '中性'
        }

        for i, raw_result in enumerate(raw_results):
            sentiment_cn = sentiment_map.get(raw_result['sentiment'], '中性')

            # Convert aspect_sentiments to Chinese
            aspect_sentiments_cn = []
            for asp_sent in raw_result.get('aspect_sentiments', []):
                aspect_sentiments_cn.append({
                    'aspect': asp_sent['aspect'],
                    'sentiment': sentiment_map.get(asp_sent['sentiment'], '中性'),
                    'sentiment_words': asp_sent['sentiment_words']
                })

            result = {
                'content': comments[i],
                'sentiment': sentiment_cn,
                'entities': raw_result['entities'],
                'sentiment_counts': raw_result['sentiment_counts'],
                'aspect_sentiments': aspect_sentiments_cn
            }
            results.append(result)

            # Statistics
            if sentiment_cn == '正面':
                stats['positive'] += 1
            elif sentiment_cn == '负面':
                stats['negative'] += 1
            else:
                stats['neutral'] += 1

        return results, stats

    except Exception as e:
        print(f"Batch analysis error: {e}")
        print("Using rule-based fallback analysis")
        # Use rule-based analysis when model fails
        results = []
        stats = {'positive': 0, 'negative': 0, 'neutral': 0}

        for comment in comments:
            result_data = rule_based_sentiment_analysis(comment)
            sentiment = result_data['sentiment']

            result = {
                'content': comment,
                'sentiment': sentiment,
                'entities': [],
                'sentiment_counts': result_data['sentiment_counts'],
                'aspect_sentiments': []
            }
            results.append(result)

            # Statistics
            if sentiment == '正面':
                stats['positive'] += 1
            elif sentiment == '负面':
                stats['negative'] += 1
            else:
                stats['neutral'] += 1

        return results, stats
