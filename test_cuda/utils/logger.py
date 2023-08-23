import sys

class OutputLogger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.logfile = logfile
    
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()  # Flush the log file after each write
    
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()