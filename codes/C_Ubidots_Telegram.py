import time
import requests
import math
import telegram_send

TOKEN = "BBFF-8oWJDDDYhli48hqJlqpXarJOf2bxyy"  # Put your TOKEN here
DEVICE_LABEL = "Monitoreo"  # Put your device label here
VARIABLE_LABEL_1 = "cumplimiento-de-distancia"  # Put your first variable label here
VARIABLE_LABEL_2 = "cumplimiento-del-barbijo"  # Put your second variable label here
VARIABLE_LABEL_3 = "numero-de-personas"  # Put your second variable label here
VARIABLE_LABEL_4 = "probabilidad-de-contagio"  # Put your second variable label here

Tiempo_de_actualizacion=60 #segundos
value_1 = 66
value_2 = 50
value_3 = 6
value_4 = 0.94
print("---------------------CORRIENDO: Ubidots.py----------------------")


def actualizar_datos(variable_1, variable_2, variable_3, variable_4):
    value_1 = variable_1
    value_2 = variable_2
    value_3 = variable_3
    value_4 = variable_4

def build_payload(variable_1, variable_2, variable_3, variable_4):
    # Creates two random values for sending data
    payload = {variable_1: value_1,
               variable_2: value_2,
               variable_3: value_3,
               variable_4: value_4}
    return payload

def post_request(payload):
    # Creates the headers for the HTTP requests
    url = "http://industrial.api.ubidots.com"
    url = "{}/api/v1.6/devices/{}".format(url, DEVICE_LABEL)
    headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}

    # Makes the HTTP requests
    status = 400
    attempts = 0
    while status >= 400 and attempts <= 5:
        req = requests.post(url=url, headers=headers, json=payload)
        status = req.status_code
        attempts += 1
        time.sleep(1)

    # Processes results
    print(req.status_code, req.json())
    if status >= 400:
        print("[ERROR] Could not send data after 5 attempts, please check \
            your token credentials and internet connection")
        return False

    print("[INFO] request made properly, your device is updated")
    return True

def main():
    payload = build_payload(VARIABLE_LABEL_1, VARIABLE_LABEL_2, VARIABLE_LABEL_3, VARIABLE_LABEL_4)
    print("[INFO] Attemping to send data")
    post_request(payload)
    telegram_send.send(messages=["ALERTA: Riesgo de propagación "+str(value_4)+"%"])
    print("[INFO] finished")

if __name__ == '__main__':
    while (True):
        main()
        time.sleep(Tiempo_de_actualizacion)
