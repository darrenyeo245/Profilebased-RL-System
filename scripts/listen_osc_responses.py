import argparse

from dotenv import load_dotenv
from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher


def handle_message(address, *args):
    print(address, args, flush=True)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Listen for RL system OSC responses.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(handle_message)
    server = osc_server.ThreadingOSCUDPServer((args.host, args.port), dispatcher)
    print(f"Listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
