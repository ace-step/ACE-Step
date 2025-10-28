from pathlib import Path

dir ='tamil_tokenizer_src'

files_name =[
    f"{dir}/tokenizer/tokenizer.py",
    f"{dir}/tokenizer/cli.py",
    f"{dir}/tokenizer/utils.py",

    f"{dir}/data/processor.py",
    f"{dir}/data/loader.py",

    f"{dir}/datasets/convert_csv.py",

    f"{dir}/logger/logger.py",
    f"{dir}/exceptions/exceptions.py",
]


for file_name in files_name:
    file_path = Path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)

print("Project structure created successfully.")


