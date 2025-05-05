// static/js/welcome.js

document.addEventListener('DOMContentLoaded', function() {
    // Check connection status with the backend
    checkConnectionStatus();

    // Function to check connection status
    function checkConnectionStatus() {
        const connectionBadgeContainer = document.getElementById('connection-badge-container');
        
        // Show checking status first
        if (connectionBadgeContainer) {
            connectionBadgeContainer.innerHTML = `
                <div class="connection-badge checking">
                    <i class="fas fa-sync fa-spin"></i> Checking connection...
                </div>
            `;
        }
        
        // Try to fetch connection status to see if the server is running
        fetch('/api/connection_status')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // If we can connect to the server, check if it's connected to the Raspberry Pi
                if (data.connected) {
                    // Add a connected badge to the UI
                    if (connectionBadgeContainer) {
                        connectionBadgeContainer.innerHTML = `
                            <div class="connection-badge connected">
                                <i class="fas fa-check-circle"></i> System Ready
                            </div>
                        `;
                    }
                } else {
                    // Add a warning badge
                    if (connectionBadgeContainer) {
                        connectionBadgeContainer.innerHTML = `
                            <div class="connection-badge warning">
                                <i class="fas fa-exclamation-circle"></i> Waiting for Backend Service
                            </div>
                        `;
                    }
                }
            })
            .catch(error => {
                console.log('Error checking connection status:', error);
                // Show error status
                if (connectionBadgeContainer) {
                    connectionBadgeContainer.innerHTML = `
                        <div class="connection-badge error">
                            <i class="fas fa-times-circle"></i> Connection Error
                        </div>
                    `;
                }
            });
    }

    // Check connection status every 5 seconds
    setInterval(checkConnectionStatus, 5000);

    // Animate feature items on hover
    const featureItems = document.querySelectorAll('.feature-item');
    featureItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            const icon = this.querySelector('.feature-icon');
            icon.style.transform = 'scale(1.2)';
            icon.style.transition = 'transform 0.3s ease';
        });
        
        item.addEventListener('mouseleave', function() {
            const icon = this.querySelector('.feature-icon');
            icon.style.transform = 'scale(1)';
        });
    });

    // Pulse animation for the start button
    const startButton = document.querySelector('.start-button');
    if (startButton) {
        startButton.addEventListener('mouseenter', function() {
            this.style.animation = 'pulse 1s infinite';
        });
        
        startButton.addEventListener('mouseleave', function() {
            this.style.animation = '';
        });
    }
});