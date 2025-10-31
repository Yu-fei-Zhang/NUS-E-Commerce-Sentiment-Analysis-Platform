// ==================== Single Analysis Page JavaScript ====================

(function() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const commentText = document.getElementById('commentText');
    const loadingMessage = document.getElementById('loadingMessage');

    // Check if we're on result page and render aspect cards
    if (window.analysisData) {
        renderAspectResults(window.analysisData);
    }

    // Exit if not on input page
    if (!analyzeBtn || !commentText) return;

    analyzeBtn.addEventListener('click', function() {
        const comment = commentText.value.trim();

        // Validate input
        if (!comment) {
            showMessage('Please enter comment content', 'error');
            return;
        }

        if (comment.length < 2) {
            showMessage('Comment must be at least 2 characters', 'error');
            return;
        }

        // Disable button, show loading
        analyzeBtn.disabled = true;
        loadingMessage.style.display = 'block';

        // Submit analysis request
        fetch('/single', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ comment: comment })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showMessage('Analysis complete! Redirecting...', 'success');
                setTimeout(() => {
                    window.location.href = '/single?id=' + data.data.analysis_id;
                }, 500);
            } else {
                throw new Error(data.message || 'Analysis failed');
            }
        })
        .catch(error => {
            console.error('Analysis error:', error);
            showMessage(error.message || 'Analysis failed, please try again', 'error');

            // Restore button state
            analyzeBtn.disabled = false;
            loadingMessage.style.display = 'none';
        });
    });

    // Submit with Enter key (optional)
    commentText.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            analyzeBtn.click();
        }
    });
})();

function renderAspectResults(data) {
    const container = document.getElementById('aspectResults');
    if (!container) return;

    const aspectSentiments = data.aspectSentiments || [];

    if (aspectSentiments.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #6b7280; padding: 2rem;">No aspects identified in the review</p>';
        return;
    }

    const sentimentLabelMap = {
        '正面': 'Positive',
        '负面': 'Negative',
        '中性': 'Neutral'
    };

    const sentimentClassMap = {
        '正面': 'positive',
        '负面': 'negative',
        '中性': 'neutral'
    };

    container.innerHTML = aspectSentiments.map(item => {
        const sentimentLabel = sentimentLabelMap[item.sentiment] || item.sentiment;
        const sentimentClass = sentimentClassMap[item.sentiment] || 'neutral';

        return `
            <div class="aspect-card">
                <div class="aspect-header">
                    <div>
                        <div class="aspect-title">Aspect</div>
                        <div class="aspect-name">${escapeHtml(item.aspect)}</div>
                    </div>
                </div>
                ${item.sentiment_words ? `
                <div class="aspect-sentiment-words">
                    <label>Sentiment Words</label>
                    <div class="sentiment-words">${escapeHtml(item.sentiment_words)}</div>
                </div>
                ` : ''}
                <div class="polarity-row">
                    <span class="polarity-label">Polarity</span>
                    <span class="polarity-badge ${sentimentClass}">
                        <span class="polarity-dot"></span>
                        ${sentimentLabel}
                    </span>
                </div>
            </div>
        `;
    }).join('');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
