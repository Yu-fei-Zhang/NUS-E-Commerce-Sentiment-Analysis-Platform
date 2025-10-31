# ==================== Flask应用主程序 ====================
from flask import Flask
from config import Config
from extensions import db, login_manager
from models import User
import os


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 初始化扩展
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'

    # 用户加载器
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # 注册蓝图
    from routes import main_bp
    app.register_blueprint(main_bp)

    # 创建必要的目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('saved_model', exist_ok=True)

    # 创建数据库表
    with app.app_context():
        db.create_all()

    return app


if __name__ == '__main__':
    app = create_app()
    print("\n" + "=" * 60)
    print("Sentiment Analysis System Starting...")
    print("=" * 60)
    print("\nPlease ensure the trained model file is placed at: saved_model/best_model.pt")
    print("\nAccess URL: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
