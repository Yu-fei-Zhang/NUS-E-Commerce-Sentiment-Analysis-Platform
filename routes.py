# ==================== 路由模块 ====================
from flask import Blueprint, render_template, redirect, url_for, request, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
import os

from extensions import db
from models import User, Analysis
from utils import allowed_file, read_file, analyze_sentiment, analyze_batch

# 创建蓝图
main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@login_required
def index():
    return redirect(url_for('main.home'))


@main_bp.route('/home')
@login_required
def home():
    return render_template('home.html')


@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username', '').strip()
            password = data.get('password', '')

            if not username or not password:
                return jsonify({'success': False, 'message': 'Please enter username and password'}), 400

            user = User.query.filter_by(username=username).first()

            if user and user.check_password(password):
                login_user(user)
                return jsonify({'success': True})
            elif not user:
                # Automatically register new user
                user = User(username=username)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                login_user(user)
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'message': 'Incorrect password'}), 401
        except Exception as e:
            print(f"Login error: {e}")
            return jsonify({'success': False, 'message': 'Server error'}), 500

    return render_template('login.html')


@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.login'))


@main_bp.route('/batch', methods=['GET', 'POST'])
@login_required
def batch():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Unsupported file format'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', f"{current_user.id}_{filename}")
        file.save(filepath)

        # Read file
        comments = read_file(filepath)
        if not comments:
            return jsonify({'success': False, 'message': 'File reading failed or no valid data'}), 400

        # Batch analysis
        results, stats = analyze_batch(comments)

        # Save analysis record
        analysis = Analysis(
            user_id=current_user.id,
            analysis_type='batch',
            filename=filename,
            total_count=len(comments),
            positive_count=stats['positive'],
            negative_count=stats['negative'],
            neutral_count=stats['neutral']
        )
        db.session.add(analysis)
        db.session.commit()

        # Save results to session
        session['batch_results'] = results

        return jsonify({'success': True, 'analysis_id': analysis.id})

    # GET request - show results
    analysis_id = request.args.get('id', type=int)
    if analysis_id:
        analysis = Analysis.query.get_or_404(analysis_id)
        if analysis.user_id != current_user.id:
            return redirect(url_for('main.batch'))
        results = session.get('batch_results', [])
        return render_template('batch.html', show_results=True, analysis=analysis, results=results)

    return render_template('batch.html', show_results=False)


@main_bp.route('/single', methods=['GET', 'POST'])
@login_required
def single():
    if request.method == 'POST':
        data = request.get_json()
        comment = data.get('comment', '').strip()

        if not comment:
            return jsonify({'success': False, 'message': 'Please enter comment content'}), 400

        # Use model for analysis
        result = analyze_sentiment(comment)

        # Save analysis record
        analysis = Analysis(
            user_id=current_user.id,
            analysis_type='single',
            content=comment,
            sentiment=result['sentiment']
        )
        analysis.aspect_sentiments = result.get('aspect_sentiments', [])
        db.session.add(analysis)
        db.session.commit()

        result['analysis_id'] = analysis.id
        return jsonify({'success': True, 'data': result})

    # GET request - show result
    analysis_id = request.args.get('id', type=int)
    if analysis_id:
        analysis = Analysis.query.get_or_404(analysis_id)
        if analysis.user_id != current_user.id:
            return redirect(url_for('main.single'))
        return render_template('single.html', show_result=True, analysis=analysis)

    return render_template('single.html', show_result=False)


@main_bp.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    type_filter = request.args.get('type', 'all')
    sentiment_filter = request.args.get('sentiment', 'all')

    query = Analysis.query.filter_by(user_id=current_user.id)

    if type_filter != 'all':
        query = query.filter_by(analysis_type=type_filter)
    if sentiment_filter != 'all':
        query = query.filter_by(sentiment=sentiment_filter)

    pagination = query.order_by(Analysis.created_at.desc()).paginate(
        page=page, per_page=10, error_out=False
    )

    # 统计数据
    total = Analysis.query.filter_by(user_id=current_user.id).count()
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    monthly = Analysis.query.filter(
        Analysis.user_id == current_user.id,
        Analysis.created_at >= month_start
    ).count()

    latest = Analysis.query.filter_by(user_id=current_user.id).order_by(
        Analysis.created_at.desc()
    ).first()

    stats = {
        'total': total,
        'monthly': monthly,
        'latest_date': latest.created_at if latest else None
    }

    return render_template('history.html',
                           analyses=pagination.items,
                           pagination=pagination,
                           stats=stats,
                           type_filter=type_filter,
                           sentiment_filter=sentiment_filter)


@main_bp.route('/history/delete/<int:id>', methods=['POST'])
@login_required
def delete_history(id):
    analysis = Analysis.query.get_or_404(id)
    if analysis.user_id != current_user.id:
        return jsonify({'success': False}), 403
    db.session.delete(analysis)
    db.session.commit()
    return jsonify({'success': True})
