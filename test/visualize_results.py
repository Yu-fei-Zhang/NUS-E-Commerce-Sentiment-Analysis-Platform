"""
Performance Test Results Visualization Tool
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from datetime import datetime
import os

# Use English only - no Chinese font issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TestReportVisualizer:
    def __init__(self, results_path: str):
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

    def plot_inference_speed(self, save_path: str = 'inference_speed.png'):
        """Plot inference speed metrics"""
        if 'inference_speed' not in self.results:
            print(f"⚠ Skip inference speed chart: no data")
            return

        speed_data = self.results['inference_speed'].get('single_inference', {})
        if not speed_data:
            print(f"⚠ Skip inference speed chart: empty data")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Time distribution
        metrics = ['mean_ms', 'median_ms', 'p95_ms', 'p99_ms']
        values = [speed_data.get(m, 0) for m in metrics]
        labels = ['Mean', 'Median', 'P95', 'P99']

        colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
        bars = ax1.bar(labels, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Inference Time Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=10)

        # Right: Redesigned box-plot style range visualization
        min_val = speed_data.get('min_ms', 0)
        max_val = speed_data.get('max_ms', 0)
        mean_val = speed_data.get('mean_ms', 0)
        median_val = speed_data.get('median_ms', mean_val)
        p25 = speed_data.get('p25_ms', min_val + (mean_val - min_val) * 0.5)
        p75 = speed_data.get('p75_ms', mean_val + (max_val - mean_val) * 0.5)

        # Create box plot style
        y_pos = 0
        box_height = 0.4

        # Main box (25%-75% percentile)
        box_width = p75 - p25
        ax2.barh([y_pos], [box_width], left=[p25], height=box_height,
                color='#64B5F6', alpha=0.7, edgecolor='#1976D2', linewidth=2, label='IQR (25%-75%)')

        # Whiskers (lines to min and max)
        ax2.plot([min_val, p25], [y_pos, y_pos], 'k-', linewidth=2, alpha=0.6)
        ax2.plot([p75, max_val], [y_pos, y_pos], 'k-', linewidth=2, alpha=0.6)

        # End caps for whiskers
        cap_height = box_height * 0.6
        ax2.plot([min_val, min_val], [y_pos - cap_height/2, y_pos + cap_height/2],
                'k-', linewidth=2, alpha=0.6)
        ax2.plot([max_val, max_val], [y_pos - cap_height/2, y_pos + cap_height/2],
                'k-', linewidth=2, alpha=0.6)

        # Median line (vertical line in the box)
        ax2.plot([median_val, median_val], [y_pos - box_height/2, y_pos + box_height/2],
                'r-', linewidth=3, label=f'Median: {median_val:.2f}ms', zorder=5)

        # Mean point (diamond marker)
        ax2.plot([mean_val], [y_pos], 'D', color='#FFA726', markersize=10,
                markeredgecolor='#F57C00', markeredgewidth=2,
                label=f'Mean: {mean_val:.2f}ms', zorder=6)

        # Add value annotations
        offset_y = box_height * 0.8
        ax2.text(min_val, y_pos - offset_y, f'{min_val:.2f}',
                ha='center', va='top', fontsize=9, fontweight='bold', color='#424242')
        ax2.text(max_val, y_pos - offset_y, f'{max_val:.2f}',
                ha='center', va='top', fontsize=9, fontweight='bold', color='#424242')

        # Calculate and display range
        range_val = max_val - min_val
        ax2.text(mean_val, y_pos + offset_y + 0.1, f'Range: {range_val:.2f}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD',
                         edgecolor='#1976D2', linewidth=1.5))

        ax2.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Time Range Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylim([-1, 1])
        ax2.set_yticks([])
        ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Inference speed chart saved: {save_path}")

    def plot_accuracy_metrics(self, save_path: str = 'accuracy_metrics.png'):
        """Plot accuracy metrics"""
        if 'accuracy' not in self.results:
            print(f"⚠ Skip accuracy chart: no accuracy data")
            print(f"   Hint: provide test_data_path when running tests")
            return False

        if 'metrics' not in self.results['accuracy']:
            print(f"⚠ Skip accuracy chart: no metrics in accuracy data")
            return False

        metrics = self.results['accuracy']['metrics']

        # Check for detailed report
        has_detailed = ('detailed_report' in self.results['accuracy'] and
                       self.results['accuracy']['detailed_report'] and
                       len(self.results['accuracy']['detailed_report']) > 0)

        # Layout based on detailed report availability
        if has_detailed:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
            print(f"ℹ Only main metrics chart (no detailed_report)")

        # Main metrics
        main_metrics = {
            'F1 Score': metrics.get('micro_f1', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0)
        }

        colors = ['#4CAF50', '#2196F3', '#FF9800']
        bars = ax1.bar(main_metrics.keys(), main_metrics.values(),
                      color=colors, alpha=0.7)
        ax1.set_ylim([0, 1.0])
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for bar, value in zip(bars, main_metrics.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=10)

        # Category F1 scores (if detailed report exists)
        if has_detailed:
            report = self.results['accuracy']['detailed_report']
            categories = []
            f1_scores = []

            for key, value in report.items():
                if key not in ['micro avg', 'macro avg', 'weighted avg'] and isinstance(value, dict):
                    if 'f1-score' in value:
                        categories.append(key)
                        f1_scores.append(value['f1-score'])

            if categories:
                y_pos = np.arange(len(categories))
                bars = ax2.barh(y_pos, f1_scores, color='#66BB6A', alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(categories, fontsize=9)
                ax2.set_xlabel('F1 Score', fontsize=12)
                ax2.set_xlim([0, 1.0])
                ax2.set_title('F1 Score by Category', fontsize=14, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)

                for bar, value in zip(bars, f1_scores):
                    width = bar.get_width()
                    ax2.text(width, bar.get_y() + bar.get_height() / 2.,
                            f'{value:.3f}',
                            ha='left', va='center', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                ax2.text(0.5, 0.5, 'No category data available',
                        ha='center', va='center', fontsize=12, transform=ax2.transAxes)
                ax2.set_title('F1 Score by Category', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Accuracy metrics chart saved: {save_path}")
        return True

    def plot_robustness_test(self, save_path: str = 'robustness_test.png'):
        """Plot robustness test results"""
        if 'robustness' not in self.results:
            print(f"⚠ Skip robustness chart: no data")
            return

        robust = self.results['robustness']
        if not robust:
            print(f"⚠ Skip robustness chart: empty data")
            return

        test_names = []
        success_status = []
        times = []

        # Translate Chinese test names to English
        name_translation = {
            '原始文本': 'Original Text',
            '短文本': 'Short Text',
            '长文本': 'Long Text',
            '特殊字符': 'Special Chars',
            '数字混合': 'Number Mix',
            '英文混合': 'English Mix',
            '空格文本': 'Spaced Text',
            '重复文本': 'Repeated Text'
        }

        for name, result in robust.items():
            # Translate name if possible
            english_name = name_translation.get(name, name)
            test_names.append(english_name)
            success_status.append(1 if result.get('success', False) else 0)
            times.append(result.get('time_ms', 0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Test pass/fail status
        colors = ['#4CAF50' if s else '#F44336' for s in success_status]
        y_pos = np.arange(len(test_names))

        bars = ax1.barh(y_pos, [1] * len(test_names), color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(test_names, fontsize=9)
        ax1.set_xlim([0, 1])
        ax1.set_xticks([])
        ax1.set_title('Robustness Test Results', fontsize=14, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', alpha=0.7, label='Pass'),
            Patch(facecolor='#F44336', alpha=0.7, label='Fail')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')

        # Right: Inference time by test case
        valid_times = [(name, time) for name, time, success in
                      zip(test_names, times, success_status) if success and time > 0]

        if valid_times:
            names, vals = zip(*valid_times)
            y_pos = np.arange(len(names))

            bars = ax2.barh(y_pos, vals, color='#2196F3', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names, fontsize=9)
            ax2.set_xlabel('Time (ms)', fontsize=12)
            ax2.set_title('Inference Time by Test Case', fontsize=14, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)

            for bar, value in zip(bars, vals):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height() / 2.,
                        f'{value:.2f}',
                        ha='left', va='center', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No timing data available',
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.set_title('Inference Time by Test Case', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Robustness test chart saved: {save_path}")

    def generate_html_report(self, save_path: str = 'performance_report.html'):
        """Generate professional HTML report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
        }}

        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 60px 80px;
            border-bottom: 4px solid #1e3c72;
        }}

        .header-content {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header h1 {{
            font-size: 2.8em;
            font-weight: 300;
            margin-bottom: 15px;
            letter-spacing: -0.5px;
        }}

        .header-subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
            font-weight: 300;
            margin-bottom: 25px;
        }}

        .header-meta {{
            display: flex;
            gap: 30px;
            font-size: 0.95em;
            opacity: 0.85;
            border-top: 1px solid rgba(255,255,255,0.2);
            padding-top: 20px;
            margin-top: 20px;
        }}

        .header-meta-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .content {{
            padding: 60px 80px;
            max-width: 1200px;
            margin: 0 auto;
        }}

        .section {{
            margin-bottom: 60px;
        }}

        .section-header {{
            border-left: 4px solid #2a5298;
            padding-left: 20px;
            margin-bottom: 35px;
        }}

        .section-header h2 {{
            font-size: 1.8em;
            font-weight: 600;
            color: #1e3c72;
            margin-bottom: 8px;
            letter-spacing: -0.3px;
        }}

        .section-header .section-subtitle {{
            font-size: 0.95em;
            color: #7f8c8d;
            font-weight: 400;
        }}

        .metrics-grid {{
            display: grid;
            gap: 25px;
            margin-top: 30px;
        }}
        
        .metrics-grid-2 {{
            grid-template-columns: repeat(2, 1fr);
        }}
        
        .metrics-grid-3 {{
            grid-template-columns: repeat(3, 1fr);
        }}
        
        .metrics-grid-4 {{
            grid-template-columns: repeat(4, 1fr);
        }}
        
        @media (max-width: 1200px) {{
            .metrics-grid-3,
            .metrics-grid-4 {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid-2,
            .metrics-grid-3,
            .metrics-grid-4 {{
                grid-template-columns: 1fr;
            }}
        }}

        .metric-card {{
            background: white;
            padding: 28px;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
            transition: all 0.3s ease;
        }}

        .metric-card:hover {{
            border-color: #2a5298;
            box-shadow: 0 4px 12px rgba(42, 82, 152, 0.1);
            transform: translateY(-2px);
        }}

        .metric-label {{
            font-size: 0.85em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 12px;
            font-weight: 600;
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: 300;
            color: #2c3e50;
            line-height: 1;
        }}

        .metric-unit {{
            font-size: 0.5em;
            color: #95a5a6;
            font-weight: 400;
            margin-left: 4px;
        }}

        .chart-container {{
            margin: 30px 0;
            background: #fafbfc;
            padding: 30px;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
        }}

        .chart-container img {{
            max-width: 100%;
            border-radius: 4px;
        }}

        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 25px;
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            overflow: hidden;
        }}

        thead {{
            background: #f8f9fa;
        }}

        th {{
            padding: 16px 20px;
            text-align: left;
            font-weight: 600;
            font-size: 0.85em;
            color: #5a6c7d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #e1e8ed;
        }}

        td {{
            padding: 16px 20px;
            border-bottom: 1px solid #f0f3f5;
            color: #2c3e50;
        }}

        tbody tr:hover {{
            background: #f8f9fa;
        }}

        tbody tr:last-child td {{
            border-bottom: none;
        }}

        .status-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
            letter-spacing: 0.3px;
        }}

        .status-pass {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}

        .status-fail {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}

        .summary-panel {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 40px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            margin-bottom: 50px;
        }}

        .summary-panel h3 {{
            font-size: 1.4em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 25px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}

        .summary-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid #dee2e6;
        }}

        .summary-item:last-child {{
            border-bottom: none;
        }}

        .summary-label {{
            font-size: 0.95em;
            color: #5a6c7d;
            font-weight: 500;
        }}

        .summary-value {{
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 40px 80px;
            text-align: center;
            border-top: 1px solid #e1e8ed;
            color: #7f8c8d;
        }}

        .footer-content {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .footer p {{
            margin: 8px 0;
            font-size: 0.9em;
        }}

        .divider {{
            height: 1px;
            background: linear-gradient(to right, transparent, #e1e8ed, transparent);
            margin: 50px 0;
        }}

        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
            .metric-card {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>Model Performance Analysis Report</h1>
                <div class="header-subtitle">BERT+CRF Aspect-Based Sentiment Analysis</div>
                <div class="header-meta">
                    <div class="header-meta-item">
                        <span>Report Generated:</span>
                        <span>{timestamp}</span>
                    </div>
                    <div class="header-meta-item">
                        <span>Framework:</span>
                        <span>PyTorch + Transformers</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="content">
            {self._generate_summary_section()}
            {self._generate_inference_speed_section()}
            {self._generate_accuracy_section()}
            {self._generate_memory_section()}
            {self._generate_robustness_section()}
            {self._generate_edge_cases_section()}
        </div>

        <div class="footer">
            <div class="footer-content">
                <p><strong>BERT+CRF Aspect-Based Sentiment Analysis Model</strong></p>
                <p>Performance Testing Framework | Automated Report Generation</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✓ HTML report saved: {save_path}")

    def _generate_summary_section(self) -> str:
        """Generate professional summary section"""
        html = '<div class="summary-panel">'
        html += '<h3>Executive Summary</h3>'
        html += '<div class="summary-grid">'
        html += '<div>'

        if 'inference_speed' in self.results:
            speed = self.results['inference_speed'].get('single_inference', {})
            mean_time = speed.get('mean_ms', 0)
            html += f'''
            <div class="summary-item">
                <span class="summary-label">Average Inference Time</span>
                <span class="summary-value">{mean_time:.2f} ms</span>
            </div>'''

        if 'accuracy' in self.results and 'metrics' in self.results['accuracy']:
            metrics = self.results['accuracy']['metrics']
            f1 = metrics.get('micro_f1', 0)
            html += f'''
            <div class="summary-item">
                <span class="summary-label">Model F1 Score</span>
                <span class="summary-value">{f1:.4f}</span>
            </div>'''

        html += '</div><div>'

        if 'memory' in self.results:
            size_mb = self.results['memory'].get('model_size_mb', 0)
            html += f'''
            <div class="summary-item">
                <span class="summary-label">Model Size</span>
                <span class="summary-value">{size_mb:.2f} MB</span>
            </div>'''

        if 'robustness' in self.results:
            robust = self.results['robustness']
            success_count = sum(1 for r in robust.values() if r.get('success', False))
            total = len(robust)
            html += f'''
            <div class="summary-item">
                <span class="summary-label">Robustness Tests Passed</span>
                <span class="summary-value">{success_count}/{total}</span>
            </div>'''

        html += '</div></div></div>'
        return html

    def _generate_inference_speed_section(self) -> str:
        """Generate professional inference speed section"""
        if 'inference_speed' not in self.results:
            return ''

        speed_data = self.results['inference_speed'].get('single_inference', {})
        if not speed_data:
            return ''

        html = '<div class="section">'
        html += '<div class="section-header">'
        html += '<h2>Inference Speed Performance</h2>'
        html += '<div class="section-subtitle">Statistical analysis of algorithm inference latency across 100 samples</div>'
        html += '</div>'
        html += '<div class="metrics-grid metrics-grid-4">'  # 4个指标，使用4列

        metrics = [
            ('Mean Latency', speed_data.get('mean_ms', 0), 'ms'),
            ('Median Latency', speed_data.get('median_ms', 0), 'ms'),
            ('95th Percentile', speed_data.get('p95_ms', 0), 'ms'),
            ('99th Percentile', speed_data.get('p99_ms', 0), 'ms')
        ]

        for label, value, unit in metrics:
            html += f'''
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.2f} <span class="metric-unit">{unit}</span></div>
            </div>
            '''

        html += '</div>'

        if os.path.exists('inference_speed.png'):
            html += '<div class="chart-container">'
            html += '<img src="inference_speed.png" alt="Inference Speed Analysis">'
            html += '</div>'

        html += '</div>'
        return html

    def _generate_accuracy_section(self) -> str:
        """Generate professional accuracy section"""
        if 'accuracy' not in self.results:
            return ''

        html = '<div class="section">'
        html += '<div class="section-header">'
        html += '<h2>Model Accuracy Metrics</h2>'
        html += '<div class="section-subtitle">Evaluation metrics on test dataset using micro-averaging</div>'
        html += '</div>'

        if 'metrics' in self.results['accuracy']:
            metrics = self.results['accuracy']['metrics']
            html += '<div class="metrics-grid metrics-grid-3">'  # 3个指标，使用3列

            acc_metrics = [
                ('F1 Score', metrics.get('micro_f1', 0)),
                ('Precision', metrics.get('precision', 0)),
                ('Recall', metrics.get('recall', 0))
            ]

            for label, value in acc_metrics:
                html += f'''
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
                '''

            html += '</div>'

        if os.path.exists('accuracy_metrics.png'):
            html += '<div class="chart-container">'
            html += '<img src="accuracy_metrics.png" alt="Model Accuracy Analysis">'
            html += '</div>'

        html += '</div>'
        return html

    def _generate_memory_section(self) -> str:
        """Generate professional memory usage section"""
        if 'memory' not in self.results:
            return ''

        mem = self.results['memory']

        html = '<div class="section">'
        html += '<div class="section-header">'
        html += '<h2>Memory Utilization</h2>'
        html += '<div class="section-subtitle">Model size and runtime memory consumption analysis</div>'
        html += '</div>'
        html += '<div class="metrics-grid metrics-grid-4">'  # 4个指标，使用4列

        mem_metrics = [
            ('Total Parameters', f"{mem.get('total_params', 0):,}", ''),
            ('Model Size', mem.get('model_size_mb', 0), 'MB'),
            ('Inference Memory', mem.get('inference_memory_mb', 0), 'MB'),
            ('Peak Memory', mem.get('peak_memory_mb', 0), 'MB')
        ]

        for label, value, unit in mem_metrics:
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)

            html += f'''
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value_str} <span class="metric-unit">{unit}</span></div>
            </div>
            '''

        html += '</div>'
        html += '</div>'
        return html

    def _generate_robustness_section(self) -> str:
        """Generate robustness test section"""
        if 'robustness' not in self.results:
            return ''

        robust = self.results['robustness']

        # Translate test names
        name_translation = {
            '原始文本': 'Original Text',
            '短文本': 'Short Text',
            '长文本': 'Long Text',
            '特殊字符': 'Special Characters',
            '数字混合': 'Number Mix',
            '英文混合': 'English Mix',
            '空格文本': 'Spaced Text',
            '重复文本': 'Repeated Text'
        }

        html = '<div class="section">'
        html += '<div class="section-header">'
        html += '<h2>Robustness Testing</h2>'
        html += '<div class="section-subtitle">Model stability across diverse input scenarios</div>'
        html += '</div>'

        html += '<table>'
        html += '<thead><tr><th>Test Scenario</th><th>Status</th><th>Latency (ms)</th><th>Output Details</th></tr></thead>'
        html += '<tbody>'

        for name, result in robust.items():
            english_name = name_translation.get(name, name)
            status = 'Pass' if result.get('success', False) else 'Fail'
            status_class = 'status-pass' if result.get('success', False) else 'status-fail'
            time_ms = result.get('time_ms', 0)

            details = ''
            if result.get('success', False):
                aspects = result.get('aspects_found', 0)
                sentiments = result.get('sentiments_found', 0)
                details = f"Aspects: {aspects}, Sentiments: {sentiments}"
            else:
                details = result.get('error', 'Unknown error')[:50]

            html += f'''
            <tr>
                <td><strong>{english_name}</strong></td>
                <td><span class="status-badge {status_class}">{status}</span></td>
                <td>{time_ms:.2f}</td>
                <td>{details}</td>
            </tr>
            '''

        html += '</tbody></table>'

        if os.path.exists('robustness_test.png'):
            html += '<div class="chart-container">'
            html += '<img src="robustness_test.png" alt="Robustness Test Chart">'
            html += '</div>'

        html += '</div>'
        return html

    def _generate_edge_cases_section(self) -> str:
        """Generate professional edge cases test section"""
        if 'edge_cases' not in self.results:
            return ''

        edge = self.results['edge_cases']

        # Translate test names
        name_translation = {
            '空文本': 'Empty Text',
            '单字': 'Single Character',
            '纯标点': 'Pure Punctuation',
            '超长文本': 'Very Long Text',
            '纯数字': 'Pure Numbers',
            '纯英文': 'Pure English',
            '特殊Unicode': 'Special Unicode',
            '换行文本': 'Newline Characters'
        }

        html = '<div class="section">'
        html += '<div class="section-header">'
        html += '<h2>Edge Case Testing</h2>'
        html += '<div class="section-subtitle">Model behavior under boundary conditions and extreme inputs</div>'
        html += '</div>'

        success_count = sum(1 for r in edge.values() if r.get('success', False))
        total_count = len(edge)

        html += f'<p style="margin-bottom: 25px; color: #5a6c7d;">'
        html += f'<strong>{success_count}</strong> out of <strong>{total_count}</strong> edge cases handled successfully'
        html += f'</p>'

        html += '<table>'
        html += '<thead><tr><th>Test Scenario</th><th>Status</th><th>Details</th></tr></thead>'
        html += '<tbody>'

        for name, result in edge.items():
            english_name = name_translation.get(name, name)
            status = 'Pass' if result.get('success', False) else 'Fail'
            status_class = 'status-pass' if result.get('success', False) else 'status-fail'

            details = ''
            if result.get('success', False):
                details = f"Results: {result.get('result_count', 0)}"
            else:
                details = result.get('error', 'Unknown error')[:50]

            html += f'''
            <tr>
                <td><strong>{english_name}</strong></td>
                <td><span class="status-badge {status_class}">{status}</span></td>
                <td>{details}</td>
            </tr>
            '''

        html += '</tbody></table>'
        html += '</div>'
        return html

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("Generating visualization reports...")
        print("="*60)

        print("\n[1/4] Generating inference speed chart...")
        self.plot_inference_speed()

        print("\n[2/4] Generating accuracy metrics chart...")
        has_accuracy = self.plot_accuracy_metrics()

        print("\n[3/4] Generating robustness test chart...")
        self.plot_robustness_test()

        print("\n[4/4] Generating HTML report...")
        self.generate_html_report()

        print("\n" + "="*60)
        print("✓ All visualization reports generated!")
        print("="*60)

        # List generated files
        print(f"\nGenerated files:")
        all_files = ['inference_speed.png', 'accuracy_metrics.png', 'robustness_test.png', 'performance_report.html']
        for filename in all_files:
            if os.path.exists(filename):
                size = os.path.getsize(filename) / 1024
                print(f"  ✓ {filename} ({size:.1f} KB)")
            else:
                print(f"  ✗ {filename} - not generated")

        if not has_accuracy:
            print(f"\n⚠ Note: accuracy_metrics.png not generated")
            print(f"   Reason: No test data provided during testing")
            print(f"   Solution: Add test_data_path='test_dataset.csv' when running tests")


def main():
    results_path = 'performance_test_report.json'

    if not os.path.exists(results_path):
        print(f"Error: Test results file not found: {results_path}")
        print("Please run model_performance_test.py first")
        return

    print(f"Reading test results: {results_path}")

    try:
        visualizer = TestReportVisualizer(results_path)
        visualizer.generate_all_visualizations()

        print(f"\n" + "="*60)
        print("Next: Open performance_report.html in your browser")
        print("="*60)
    except Exception as e:
        print(f"\nError: Failed to generate visualizations")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()