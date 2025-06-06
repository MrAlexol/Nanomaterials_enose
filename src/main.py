# -*- coding: utf-8 -*-
from gui.gui import MainGui


def main():
    app = MainGui()
    app.protocol("WM_DELETE_WINDOW", app.quit)
    app.mainloop()

if __name__ == "__main__":
    main()
