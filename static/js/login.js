// ==================== Login Page JavaScript ====================

document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');

    if (!loginForm) return;

    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value;

        if (!username || !password) {
            showMessage('Please enter username and password', 'error');
            return;
        }

        // Submit login request
        fetch(loginForm.action || '/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showMessage('Login successful!', 'success');
                setTimeout(() => {
                    window.location.href = '/home';
                }, 500);
            } else {
                showMessage(data.message || 'Login failed', 'error');
            }
        })
        .catch(error => {
            console.error('Login error:', error);
            showMessage('Network error, please try again', 'error');
        });
    });
});
