import time

class colorpalette:

    OK = '\033[1;92m' #GREEN
    WARNING = '\033[1;93m' #YELLOW
    FAIL = '\033[1;91m' #RED
    INFO = '\033[1;94m' #BLUE
    RESET = '\033[0m' #RESET COLOR
    
class show_config(colorpalette):
    
    def __init__(
        self,
        titles,
    ):
        self.filepath = f"config{show_config.timeRecorder()}.txt"
        self.titles = titles       
        
    def log_every(self):
        for title, content in self.titles.items():
            show_config.log(self.filepath, self.INFO, self.RESET, title, content)
        
    @classmethod    
    def log(cls, file, color, reset, title, text):
        with open(file, 'a') as f:
            f.write(f"{color}{title}{reset}: {text}\n")
            
    @staticmethod
    def timeRecorder():
        return time.strftime("%Y%m%d-%H%M%S",time.localtime())
            
    def __repr__(self):
        with open(self.filepath, 'r') as f:
            content = f.read()
        return content