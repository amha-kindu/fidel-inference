workers = 1
bind = "0.0.0.0:7890"
accesslog = "-"
errorlog = "-"
loglevel = "info"
worker_tmp_dir = "/dev/shm"
# Tune keepalive to play nice with SSE
keepalive = 75
