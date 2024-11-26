from enum import Enum

class JobStatus(Enum):
    WaitingQueue = "waiting_queue"
    Processing = "processing"
    Done = "done"
    Failed = "failed"
