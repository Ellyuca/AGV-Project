
class Log(object):

    def __init__(self, path, append=False):
        self.f_log = open(path,"w" if not append else "a+")
    
    def log(self, *args):
        self.f_log.write(*args)
        self.f_log.write("\n")
        self.f_log.flush()

    def close(self):
        self.f_log.close()

    def __del__(self):
        self.f_log.close()
