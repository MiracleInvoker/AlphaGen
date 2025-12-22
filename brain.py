import json
from datetime import timedelta
from os import getenv
from time import sleep
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv, set_key
from rich.console import Console

console = Console()


class API:
    base = "https://api.worldquantbrain.com"
    auth = base + "/authentication"
    simul = base + "/simulations"
    alpha = base + "/alphas/"
    alphas = base + "/users/self/alphas"
    data_field = base + "/data-fields/"

    def recordsets(alpha_id, recordset):
        return API.alpha + alpha_id + "/recordsets/" + recordset

    def performance(alpha_id, challenge):
        return (
            API.base
            + "/competitions/"
            + challenge
            + "/alphas/"
            + alpha_id
            + "/before-and-after-performance"
        )


def session_information(brain_session):
    response = brain_session.get(API.auth)

    if response.status_code == requests.status_codes.codes.ok:
        resp_json = response.json()

        user_id = resp_json["user"]["id"]
        ttl = int(resp_json["token"]["expiry"])

        return user_id, ttl

    return None, 0


def login():
    console.print("Logging In...", style="yellow")

    brain_session = requests.Session()
    brain_session.auth = (getenv("email"), getenv("password"))
    response = brain_session.post(API.auth)

    if response.status_code == requests.status_codes.codes.unauthorized:
        if response.headers["WWW-Authenticate"] == "persona":
            input(
                "Complete Biometrics Authentication and press any key to continue: "
                + urljoin(response.url, response.headers["Location"])
            )
            response = brain_session.post(
                urljoin(response.url, response.headers["Location"])
            )

            set_key(".env", "persona", "true")

        else:
            console.print("Incorrect Email and Password.", style="red")
            return None

    else:
        set_key(".env", "persona", "false")

    headers = response.headers
    t = headers["Set-Cookie"].split(";")[0][2:]
    set_key(".env", "t", t)
    brain_session.cookies.update({"t": t})

    data = response.json()
    user_id = data["user"]["id"]
    token_expiry = int(data["token"]["expiry"])
    console.print(
        f"{user_id} Logged In. | TTL: {str(timedelta(seconds=token_expiry))}",
        style="yellow",
    )

    return brain_session


def re_login():
    load_dotenv()
    t = getenv("t")

    brain_session = requests.Session()
    brain_session.cookies.update({"t": t})

    user_id, token_expiry = session_information(brain_session)

    if token_expiry < 7200:
        return login()

    console.print(
        f"{user_id} Logged In. | TTL: {str(timedelta(seconds=token_expiry))}",
        style="yellow",
    )

    return brain_session


class Alpha:

    def simulate(brain_session, simulation_data):

        simulation_response = brain_session.post(API.simul, json=simulation_data)
        simulation_progress_url = simulation_response.headers["Location"]

        trial = 0
        while True:
            try:
                simulation_progress = brain_session.get(simulation_progress_url)
                trial += 1

            except Exception as e:
                console.print(e, style="red")
                continue

            try:
                simulation_response = simulation_progress.json()

            except requests.JSONDecodeError:
                continue

            if "alpha" in simulation_response:
                break

            if "progress" not in simulation_response:
                continue

            progress = simulation_response["progress"]

            print("\r\x1b[2K", end="")
            console.print(
                f"Attempt #{trial} | Simulation Progress: {int(100 * progress)}%",
                end="",
                style="yellow",
            )
            sleep(float(simulation_progress.headers["Retry-After"]))

        alpha_id = simulation_response["alpha"]
        print("\r\x1b[2K", end="")
        console.print(f"Attempt #{trial} | Alpha ID: {alpha_id}", style="yellow")

        return alpha_id

    def simulation_result(brain_session, alpha_id):

        while True:
            try:
                simulation_result = brain_session.get(API.alpha + alpha_id)

            except Exception as e:
                console.print(e, style="red")
                continue

            try:
                result_json = simulation_result.json()
                break

            except requests.JSONDecodeError:
                pass

            sleep(float(simulation_result.headers["Retry-After"]))

        return result_json

    def recordsets(brain_session, alpha_id, recordset):

        trial = 0
        while True:
            try:
                response = brain_session.get(API.recordsets(alpha_id, recordset))
                trial += 1

            except Exception as e:
                console.print(e, style="red")
                continue

            try:
                resp_json = response.json()
                break

            except requests.exceptions.JSONDecodeError:
                pass

            print("\r\x1b[2K", end="")
            console.print(
                f"Attempt #{trial} | retrieving {recordset} chart...",
                end="",
                style="yellow",
            )

            sleep(float(response.headers["Retry-After"]))

        print("\r\x1b[2K", end="")
        console.print(
            f"Attempt #{trial} | {recordset} chart retrieved.", style="yellow"
        )

        resp_data = resp_json["records"]
        return resp_data

    def performance(brain_session, alpha_id, challenge):

        while True:
            try:
                performance_result = brain_session.get(
                    API.performance(alpha_id, challenge)
                )

            except Exception as e:
                console.print(e, style="red")
                continue

            try:
                performance_json = performance_result.json()
                break

            except requests.exceptions.JSONDecodeError:
                pass

            sleep(float(performance_result.headers["Retry-After"]))

        score = performance_json["score"]["after"] - performance_json["score"]["before"]
        return score


def extract_alphas(brain_session, submitted=True, conditions={}):
    alphas = []

    payload = {"limit": 100, "offset": 0, "order": "-dateCreated", "hidden": "false"}

    payload.update(conditions)

    if submitted:
        payload["status!"] = "UNSUBMITTED"
        file_name = "submitted_alphas.json"

    else:
        payload["status"] = "UNSUBMITTED"
        file_name = "unsubmitted_alphas.json"

    while True:
        resp = brain_session.get(API.alphas, params=payload)
        resp_json = resp.json()

        alphas += resp_json["results"]
        console.print(f"{len(alphas)} Alphas Extracted...", style="yellow")

        if resp_json["next"] == None:
            break

        else:
            payload["offset"] += 100

    console.print(f"Total Alphas Extracted: {len(alphas)}", style="green")

    with open(file_name, "w") as f:
        json.dump(alphas, f, indent=2)


def data_field(brain_session, data_field):
    resp = brain_session.get(API.data_field + data_field)
    resp_json = resp.json()

    return resp_json["type"], resp_json["description"]
