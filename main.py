import os
import threading

from dotenv import load_dotenv

from osc import OSCInterface
from profile_system import RLProfileSystem
from storage import MediaIPStorageClient


def main() -> None:
    load_dotenv()

    storage_client = MediaIPStorageClient.from_env()
    rl_system = RLProfileSystem(mediaip_storage_client=storage_client)
    osc = OSCInterface(rl_system=rl_system, auto_start=True)

    try:
        threading.Event().wait()
    finally:
        osc.shutdown()


if __name__ == "__main__":
    main()
