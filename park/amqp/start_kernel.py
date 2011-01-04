from park.environment import Environment
from park.store import Store
from .config import SERVICE_HOST
from .core import connect, start_kernel

def start(jobid, module, init):
    server = connect(SERVICE_HOST)
    env = Environment(jobid=jobid, store=Store())
    kernel = env.get_kernel(module, init)
    start_kernel(server, jobid, kernel)

if __name__ == "__main__":
    import sys
    start(*sys.argv[1:])