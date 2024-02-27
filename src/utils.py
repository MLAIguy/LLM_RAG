import json
import boto3


def read_json_to_dict(path: str) -> dict:
    try:
        with open(path) as json_file:
            json_dict = json.load(json_file)
            return json_dict
    except Exception as e:
        raise f"During reading of json file {json_file} an error occur during reading : {e}"


def get_boto3_client(service_name):
    return boto3.client(service_name=service_name)
