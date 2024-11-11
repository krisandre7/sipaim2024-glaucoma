from enum import Enum

class Task(Enum):
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    MULTILABEL = 'multilabel'

class Stage(Enum):
    FIT = 'fit'
    TEST = 'test'
    PRED = 'pred'
    
class TrainStage(Enum):
    TRAIN = 'train'
    VALIDATE = 'val'
    
class JustRAIGSTask(Enum):
    REFERRAL = 'referral'
    JUSTIFICATION = 'justification'