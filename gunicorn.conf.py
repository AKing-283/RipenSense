# Number of worker processes
workers = 3

# Binding to the specified host and port
bind = "0.0.0.0:8080"  # Render will use port 8080

# Timeout for workers
timeout = 120  # Increase timeout if needed for long processes

# Log file settings (optional)
accesslog = '-'
errorlog = '-'

# Specify the worker class (sync is default, but you can use async or gevent)
worker_class = 'sync'  # Can be 'gevent' for asynchronous requests

# Enable/disable keep-alive (default is True)
keepalive = 5
