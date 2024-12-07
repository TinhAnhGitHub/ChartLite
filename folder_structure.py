import os



def draw_folder_structure(base_dir, ignore_patterns = None, prefix = ""):
    ignore_patterns = ignore_patterns or []

    entries = [e for e in os.listdir(base_dir) if not any(p in e for p in ignore_patterns)]
    entries.sort()

    for i,entry in enumerate(entries):
        path = os.path.join(base_dir, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        
        print(f"{prefix}{connector}{entry}")

        if os.path.isdir(path):
            new_prefix = prefix + ("    " if i == len(entries) - 1 else "│   ")
            draw_folder_structure(path, ignore_patterns, new_prefix)


base_directory = "."  # Change to your target directory
ignore_list = ["__pycache__", ".git", ".DS_Store", "node_modules"]
draw_folder_structure(base_directory, ignore_list)
