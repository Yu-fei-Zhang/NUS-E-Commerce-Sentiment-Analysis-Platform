// ==================== Single Analysis Page JavaScript ====================

(function() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const commentText = document.getElementById('commentText');
    const loadingMessage = document.getElementById('loadingMessage');

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
