import time
import requests
import telegram_send

TOKEN = "BBFF-8oWJDDDYhli####################"  # Put your TOKEN here
DEVICE_LABEL = "Monitoreo"  # Put your device label here
VARIABLE_LABEL_1 = "cumplimiento-de-distancia"  # Put your first variable label here
VARIABLE_LABEL_2 = "cumplimiento-del-barbijo"  # Put your second variable label here
VARIABLE_LABEL_3 = "numero-de-personas"  # Put your second variable label here
VARIABLE_LABEL_4 = "probabilidad-de-contagio"  # Put your second variable label here

Tiempo_de_actualizacion=60 #segundos
#Valores de prueba
value_1 = 0
value_2 = 66.6
value_3 = 3
value_4 = 16.76
print("---------------------CORRIENDO: Ubidots.py----------------------")

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
if __name__ == '__main__':
    while (True):
        payload = {VARIABLE_LABEL_1: value_1,
                   VARIABLE_LABEL_2: value_2,
                   VARIABLE_LABEL_3: value_3,
                   VARIABLE_LABEL_4: value_4}
        print("[INFO] Attemping to send data")
        post_request(payload)
        telegram_send.send(messages=["ALERTA: Riesgo de propagación " + str(value_4) + "%"])
        print("[INFO] finished")
        time.sleep(Tiempo_de_actualizacion)