from argparse import ArgumentParser
from src.utils import read_json_to_dict


def get_input_configs() -> dict:
    args = ArgumentParser(
        description="PyTorch Natural Language Processing Template"
    )

    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)"
    )

    rag_kwargs = vars(args.parse_args())
    config_path = rag_kwargs.get('config', "")

    if config_path:
        input_config = read_json_to_dict(config_path)
    else:
        raise Exception('Config file needed')

    return input_config
