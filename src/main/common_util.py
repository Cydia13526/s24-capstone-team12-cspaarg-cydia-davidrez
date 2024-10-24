from cryptography.fernet import Fernet

def get_fred_api_key():
    with open("src/resources/keys/fred_api_key.txt", "rb") as file:
        key = file.read()
    cipher_suite = Fernet(key)
    with open("src/resources/keys/fred_api_key_encrypt.txt", "rb") as file:
        encrypted_key = file.read()
    decrypted_key = cipher_suite.decrypt(encrypted_key).decode()
    return decrypted_key