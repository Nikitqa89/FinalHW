import re

INPUT_FILE = 'data.txt'
OUTPUT_FILE = 'filtered_poetry_lines.txt'

# Ограниченный набор символов (~70)
ALLOWED_CHARS = set(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    " .,!?;:—«»'\"…-"
)

def is_poetry_line(line: str) -> bool:
    # 1. Строка должна начинаться с табуляции
    if not line.startswith('\t'):
        return False

    stripped = line.strip()

    # 2. Длина от 10 до 60 символов
    if not (10 <= len(stripped) <= 60):
        return False

    # 3. Исключаем строки с латиницей
    if re.search(r'[a-zA-Z]', stripped):
        return False

    # 4. Убираем строки с 2+ заглавных русских букв подряд
    if re.search(r'[А-ЯЁ]{2,}', stripped):
        return False

    return True

def clean_line(line: str) -> str:
    # Удаляем примечания в квадратных скобках
    line = re.sub(r'\[.*?\]', '', line)

    # Удаляем неразрывный пробел
    line = line.replace('\u00A0', ' ')

    # Удаляем все символы, кроме разрешённых
    line = ''.join(ch for ch in line if ch in ALLOWED_CHARS or ch == '\t')

    # Удаляем множественные пробелы
    line = re.sub(r'\s{2,}', ' ', line)

    return line.strip()

def preprocess_poetry_file():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    poetry_lines = []
    for line in lines:
        if is_poetry_line(line):
            cleaned = clean_line(line)
            poetry_lines.append(cleaned)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(poetry_lines))

    print(f"✅ Сохранено {len(poetry_lines)} строк в {OUTPUT_FILE}")

if __name__ == '__main__':
    preprocess_poetry_file()