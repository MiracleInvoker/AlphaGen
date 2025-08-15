from datetime import timedelta
from dotenv import load_dotenv, set_key
import json
from os import getenv
import requests
from rich.console import Console
from time import sleep
from urllib.parse import urljoin
from utils import terminal


class API:
    base = "https://api.worldquantbrain.com"
    auth = base + "/authentication"
    simul = base + "/simulations"
    alpha = base + "/alphas/"
    alphas = base + "/users/self/alphas"
    data_field = base + "/data-fields/"

    def pnl(alpha_id):
        return API.alpha + alpha_id + "/recordsets/pnl"


console = Console()


def login():
    load_dotenv()
    t = getenv("t")

    brain_session = requests.Session()
    brain_session.cookies.update({
        "t": t
    })

    response = brain_session.get(API.auth)
    
    if (response.status_code == requests.status_codes.codes.no_content):
        console.print("Logging In...", style = 'yellow')

        brain_session.auth = (getenv("email"), getenv("password"))
        response = brain_session.post(API.auth)

        if response.status_code == requests.status_codes.codes.unauthorized:
            if response.headers["WWW-Authenticate"] == "persona":
                input("Complete Biometrics Authentication and press any key to continue: " + urljoin(response.url, response.headers["Location"]))

                response = brain_session.post(urljoin(response.url, response.headers["Location"]))

            else:
                console.print("Incorrect Email and Password.", style = 'red')

                return None

        headers = response.headers

        t = headers["Set-Cookie"].split(";")[0][2:]
        set_key(".env", "t", t) 

    brain_session.cookies.update({
        "t": t
    })

    data = response.json()

    user_id = data["user"]["id"]
    token_expiry = int(data["token"]["expiry"])

    console.print(f"{user_id} Logged In. | TTL: {str(timedelta(seconds = token_expiry))}", style = 'yellow')

    return brain_session


class Alpha:

    def simulate(brain_session, simulation_data):

        simulation_response = brain_session.post(API.simul, json = simulation_data)
        simulation_progress_url = simulation_response.headers["Location"]

        trial = 0
        while True:
            simulation_progress = brain_session.get(simulation_progress_url)
            trial += 1

            try:
                simulation_response = simulation_progress.json()

            except requests.JSONDecodeError:
                continue

            if ("alpha" in simulation_response):
                break

            if ("progress" not in simulation_response):
                continue

            progress = simulation_response["progress"]

            terminal.clear_line()
            console.print(f"Attempt #{trial} | Simulation Progress: {int(100 * progress)}%", end = '', style = 'yellow')
            sleep(float(simulation_progress.headers["Retry-After"]))

        alpha_id = simulation_response["alpha"]
        terminal.clear_line()
        console.print(f"Attempt #{trial} | Alpha ID: {alpha_id}", style = 'yellow')

        return alpha_id
    
    def simulation_result(brain_session, alpha_id):
        
        while True:
            simulation_result = brain_session.get(API.alpha + alpha_id)

            if simulation_result.text:
                break

            sleep(float(simulation_result.headers["Retry-After"]))

        result_json = simulation_result.json()
        return result_json
    
    def pnl(brain_session, alpha_id):

        trial = 0
        while True:
            pnl = brain_session.get(API.pnl(alpha_id))
            trial += 1

            if (pnl.text):
                break

            terminal.clear_line()
            console.print(f"Attempt #{trial} | Retrieving PnL Chart...", end = '', style = 'yellow')

            sleep(float(pnl.headers["Retry-After"]))

        terminal.clear_line()
        console.print(f"Attempt #{trial} | PnL Chart Retrieved.", style = 'yellow')

        pnl_json = pnl.json()
        pnl_data = pnl_json["records"]
        return pnl_data


def extract_alphas(brain_session, submitted = True, conditions = {}, file_name = None):
    alphas = []

    payload = {
        'limit': 100,
        'offset': 0,
        'order': '-dateCreated',
        'hidden': 'false'
    }

    payload.update(conditions)

    if (file_name == None):
        if (submitted):
            payload['status!'] = 'UNSUBMITTED'
            file_name = "submitted_alphas.json"
        else:
            payload['status'] = 'UNSUBMITTED'
            file_name = "unsubmitted_alphas.json"

    while (True):
        resp = brain_session.get(API.alphas, params = payload)
        resp_json = resp.json()

        alphas += resp_json['results']
        console.print(f"{len(alphas)} Alphas Extracted...", style = 'yellow')

        if (resp_json['next'] == None):
            break
        else:
            payload['offset'] += 100

    console.print(f"Total Alphas Extracted: {len(alphas)}", style = 'green')

    with open(file_name, 'w') as f:
        json.dump(alphas, f, indent = 2)


def data_field(brain_session, data_field):
    resp = brain_session.get(API.data_field + data_field)
    resp_json = resp.json()

    return resp_json['description']