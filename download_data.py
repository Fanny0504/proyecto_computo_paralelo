import os
import sys
import zipfile
import gdown

DRIVE_FILE_ID = "1t_osx9vQ6vWACgbQfTS8tTXLpkJTmoU2"

# Rutas locales
TARGET_DIR = os.path.join("data", "full_dataset")
ZIP_DESTINATION = os.path.join("data", "dataset_temp.zip")


def download_and_extract_dataset():
    # 1. Verificar si el dataset ya existe para evitar descargas redundantes
    if os.path.exists(TARGET_DIR) and len(os.listdir(TARGET_DIR)) > 0:
        total_files = len(os.listdir(TARGET_DIR))
        print(f"[OK] El directorio '{TARGET_DIR}' ya existe y contiene {total_files} elementos.")
        print("No se requiere descarga.")
        return

    # 2. Asegurar que el directorio contenedor exista
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("Iniciando descarga del dataset desde Google Drive...")
    print("=" * 60)

    # 3. Descarga mediante gdown
    download_url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    
    try:
        # fuzzy=True permite a gdown resolver URLs completas o IDs directos
        output = gdown.download(download_url, ZIP_DESTINATION, quiet=False, fuzzy=True)
        
        if not output or not os.path.exists(ZIP_DESTINATION):
            print("\n[ERROR] Falló la descarga. Verifica que el archivo de Drive tenga permisos de lectura públicos:")
            print("        'Cualquier persona con el enlace puede ver'.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un fallo durante la descarga: {e}")
        sys.exit(1)

    # 4. Descompresión con zipfile
    print("\n" + "=" * 60)
    print("Descomprimiendo archivos...")
    print("=" * 60)

    try:
        with zipfile.ZipFile(ZIP_DESTINATION, "r") as zip_ref:
            # Lista de archivos para reporte
            file_list = zip_ref.namelist()
            print(f"Extrayendo {len(file_list)} archivos en '{TARGET_DIR}'...")
            zip_ref.extractall(TARGET_DIR)
            
        print("[OK] Extracción completada exitosamente.")

    except zipfile.BadZipFile:
        print("\n[ERROR] El archivo descargado no es un ZIP válido o la descarga quedó incompleta.")
        sys.exit(1)
        
    finally:
        # 5. Limpieza del ZIP temporal para liberar espacio en disco
        if os.path.exists(ZIP_DESTINATION):
            print("Eliminando archivo temporal comprimido...")
            os.remove(ZIP_DESTINATION)
            print("[OK] Limpieza terminada.")

    print("\nDataset listo para ejecución en tus pipelines de PCA.")


if __name__ == "__main__":
    download_and_extract_dataset()