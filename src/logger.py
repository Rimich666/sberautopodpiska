from datetime import datetime
import logging
from colorama import init, Fore, Style

# Инициализация colorama
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Кастомный форматтер для цветного вывода логов"""
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, Fore.WHITE)
        levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"
        message = super().format(record)
        return f"{Fore.WHITE}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL} | {levelname} | {message}"


def setup_logging():
    """Настройка логгера с цветным выводом"""
    logger = logging.getLogger()

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(message)s'))

    logger.addHandler(handler)
    return logger


logger = setup_logging()
