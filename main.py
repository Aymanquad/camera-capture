from colorama import init as colorama_init
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Launch Tkinter UI
from ui_tk import run as run_ui


def main():
    colorama_init(autoreset=True)
    run_ui()


if __name__ == "__main__":
    main()
