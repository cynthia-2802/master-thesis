from dotenv import load_dotenv

from .config import Config
from .orchestration import run_pipeline


def main() -> None:
    load_dotenv()
    run_pipeline(Config.from_env())


if __name__ == "__main__":
    main()
