from os import system


class terminal:
    def clear_line():
        print('\r\x1b[2K', end = '')

    def clear():
        system('cls')