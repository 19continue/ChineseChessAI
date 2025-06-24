import os
import time
import fcntl
import errno
import atexit

class FileLock:
    def __init__(self, filename):
        self.filename = filename
        self.lockfile = filename + ".lock"
        self.lock_fd = None
        
    def acquire(self, timeout=30):
        start_time = time.time()
        while True:
            try:
                self.lock_fd = os.open(self.lockfile, os.O_CREAT | os.O_RDWR)
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                atexit.register(self.release)
                return True
            except IOError as e:
                if e.errno != errno.EAGAIN:
                    raise
                if time.time() - start_time > timeout:
                    return False
                time.sleep(0.1)
                
    def release(self):
        if self.lock_fd is not None:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            os.close(self.lock_fd)
            try:
                os.remove(self.lockfile)
            except OSError:
                pass
            self.lock_fd = None
            
    def __enter__(self):
        if not self.acquire():
            raise TimeoutError("Could not acquire lock")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release() 