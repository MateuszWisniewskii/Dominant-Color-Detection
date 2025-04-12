file_path = "../test_files/obrazek1.png"

#file_path = input("Podaj nazwę pliku wraz z rozszerzeniem i ścieżką: ")

try:
    file_handle = open(file_path, "rb")
    print("Udało się otworzyć plik :)")
except:
    print(f"Nie odnaleziono pliku: {file_path}")
    print("Sprawdź jeszcze raz ścieżkę do pliku i uruchom program na nowo") 

for line in file_handle:
    print(line)

print("=======================================================")

