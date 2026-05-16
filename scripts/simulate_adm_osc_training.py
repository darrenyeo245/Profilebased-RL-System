import argparse
import math
import time

from dotenv import load_dotenv
from pythonosc import udp_client


def _actor_position(elapsed: float) -> list[float]:
    x = math.sin(elapsed * 0.7)
    y = math.cos(elapsed * 0.5)
    z = math.sin(elapsed * 0.3) * math.cos(elapsed * 0.2)
    return [x, y, z]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Continuously send simulated ADM-OSC actor positions to the RL system."
    )
    parser.add_argument("--host", default="127.0.0.1", help="RL system OSC host")
    parser.add_argument("--port", type=int, default=9001, help="RL system OSC port")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds. Use 0 to run until Ctrl+C.")
    parser.add_argument("--hz", type=float, default=20.0, help="Actor signal frequency")
    args = parser.parse_args()

    if args.hz <= 0:
        raise ValueError("--hz must be positive")

    client = udp_client.SimpleUDPClient(args.host, args.port)
    start_time = time.monotonic()
    period = 1.0 / args.hz

    print(f"[start] Sending /adm/obj/101/xyz to {args.host}:{args.port} at {args.hz} Hz", flush=True)

    try:
        next_tick = time.monotonic()
        while True:
            now = time.monotonic()
            elapsed = now - start_time
            if args.duration > 0 and elapsed >= args.duration:
                break

            position = _actor_position(elapsed)
            client.send_message("/adm/obj/101/xyz", position)
            print(f"[actor] /adm/obj/101/xyz {position}", flush=True)

            next_tick += period
            sleep_time = max(0.0, next_tick - time.monotonic())
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("[stop] Interrupted by user", flush=True)

    print("[done] Actor simulation finished", flush=True)


if __name__ == "__main__":
    main()
