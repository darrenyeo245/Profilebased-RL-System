import argparse

from dotenv import load_dotenv
from pythonosc import udp_client


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Send a local OSC command to the RL system.")
    parser.add_argument("address", help="OSC address, e.g. /rl/set/config/setup/1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.host, args.port)
    client.send_message(args.address, [])
    print(f"Sent {args.address} to {args.host}:{args.port}")


if __name__ == "__main__":
    main()
