from subprocess import call

def start():
    call('pip install -r requirements.txt', shell=True)
    call('streamlit run src/main/app.py', shell=True)

if __name__ == "__main__":
    start()