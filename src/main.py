from src.config_parser import get_input_configs
from src.StreamLitWrapper import StreamLitWrapper


def main():
    input_config = get_input_configs()
    stream_lit = StreamLitWrapper(**input_config['streamlit'])
    stream_lit.run(input_config['data_config'], model_config=input_config['model_config'])
    print('Done!')


if __name__ == '__main__':
    main()
