# ==================== 数据库模型 ====================
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Analysis(db.Model):
    __tablename__ = 'analyses'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    analysis_type = db.Column(db.String(20), nullable=False)  # 'single' or 'batch'
    content = db.Column(db.Text)  # 对于单条分析存储评论内容
    sentiment = db.Column(db.String(20))  # 情感: 正面/负面/中性

    # 批量分析统计
    total_count = db.Column(db.Integer, default=1)
    positive_count = db.Column(db.Integer, default=0)
    negative_count = db.Column(db.Integer, default=0)
    neutral_count = db.Column(db.Integer, default=0)

    filename = db.Column(db.String(200))  # 上传的文件名
    aspect_sentiments_json = db.Column(db.Text)  # JSON格式存储aspect-sentiment pairs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def aspect_sentiments(self):
        """解析JSON字符串为列表"""
        if self.aspect_sentiments_json:
            import json
            try:
                return json.loads(self.aspect_sentiments_json)
            except:
                return []
        return []

    @aspect_sentiments.setter
    def aspect_sentiments(self, value):
        """将列表序列化为JSON字符串"""
        import json
        self.aspect_sentiments_json = json.dumps(value, ensure_ascii=False) if value else None
