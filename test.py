from config import get_args
import time

args = get_args()
for i in range(args.epochs):
    if (i+1) % args.print_freq == 0:
        print("asd")
    time.sleep(0.5)