from tkinter import ttk, filedialog
import os

class FileListPanel(ttk.Frame):
    def __init__(self, parent, files_owner, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.MAX_VISIBLE_ROWS = 6
        self.files = []
        self.files_owner = files_owner

        self.tree_frame = ttk.Frame(self)
        self.tree_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(
            self.tree_frame,
            columns=("name", "type", "size"),
            show="headings",
            selectmode="browse",
            height=self.MAX_VISIBLE_ROWS
        )
        self.tree.heading("name", text="Файл")
        self.tree.heading("type", text="Тип")
        self.tree.heading("size", text="Размер")
        self.tree.column("name", anchor="w", width=200)
        self.tree.column("type", anchor="center", width=80)
        self.tree.column("size", anchor="center", width=80)

        self.scrollbar = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.tree_frame.columnconfigure(0, weight=1)
        self.tree_frame.rowconfigure(0, weight=1)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=5)

        left_btns = ttk.Frame(btn_frame)
        left_btns.pack(side="left")

        add_btn = ttk.Button(left_btns, text="+", width=3, command=self.add_file)
        add_btn.pack(side="left", padx=(0, 5))

        del_btn = ttk.Button(left_btns, text="–", width=3, command=self.remove_selected)
        del_btn.pack(side="left")

        clear_btn = ttk.Button(btn_frame, text="Очистить", command=self.remove_all)
        clear_btn.pack(side="right", padx=(5, 0))

        self.update_scrollbar_visibility()

    def update_scrollbar_visibility(self):
        row_count = len(self.tree.get_children())
        if row_count > self.MAX_VISIBLE_ROWS:
            self.scrollbar.grid()
        else:
            self.scrollbar.grid_remove()

    def add_file(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Файлы CSV", "*.csv"), ("Все файлы", "*.*")])
        for file_path in file_paths:
            if file_path and file_path not in self.files:
                name = os.path.basename(file_path)
                file_type = os.path.splitext(file_path)[1].lstrip('.').lower() or "неизвестно"
                size = format_size(os.path.getsize(file_path))
                self.files.append(file_path)
                self.tree.insert("", "end", values=(name, file_type, size))
                self.files_list_updated()
        self.update_scrollbar_visibility()

    def remove_selected(self):
        selected_item = self.tree.selection()
        if selected_item:
            index = self.tree.index(selected_item[0])
            self.tree.delete(selected_item[0])
            del self.files[index]
            self.files_list_updated()
        self.update_scrollbar_visibility()
    
    def remove_all(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.files.clear()
        self.files_list_updated()
        self.update_scrollbar_visibility()

    def get_files(self):
        return self.files
    
    def files_list_updated(self):
        self.files_owner.set_selected_files(self.files)

def format_size(bytes_size: int) -> str:
    for unit in ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} ПБ"
