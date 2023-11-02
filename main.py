from src.main_init import main_init, main
from src.web import flask_thread_start

if __name__ == "__main__":
    args = main_init()
    flask_thread_start(args)
    main()
