import logging
import torch.distributed as dist

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

class RankFilter(logging.Filter):
    def __init__(self, rank0_only=True):
        super().__init__()
        self.rank0_only = rank0_only

    def filter(self, record):
        record.rank = get_rank()
        return not self.rank0_only or record.rank == 0

def _setup_logger(name, rank0_only):
    l = logging.getLogger(name)
    l.setLevel(logging.INFO)
    if not l.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(RankFilter(rank0_only))
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [Rank %(rank)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        l.addHandler(handler)
        l.propagate = False
    return l

logger = _setup_logger("rank0", rank0_only=True)
all_rank_logger = _setup_logger("allrank", rank0_only=False)