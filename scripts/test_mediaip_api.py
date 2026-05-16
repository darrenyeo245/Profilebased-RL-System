from dotenv import load_dotenv

from profile_system import RLProfileSystem
from storage import MediaIPStorageClient


def main() -> None:
    load_dotenv()

    setup_id = input("Setup-ID zum Testen (z.B. 1): ").strip() or "1"
    client = MediaIPStorageClient.from_env()
    system = RLProfileSystem(mediaip_storage_client=client)
    status = system.load_setup(setup_id)

    print("Setup geladen:")
    print(status.to_dict())
    print(f"Cache-Pfad: {client.paths.rl_system_root}")


if __name__ == "__main__":
    main()
