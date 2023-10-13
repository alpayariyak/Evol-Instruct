import argparse
from src.evol_instruct import EvolInstruct
from src.utils import load_dataset, save_dataset, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--evol_model", type=str, default="gpt-4")
    parser.add_argument("--eliminator_model", type=str, default="gpt-35-turbo")
    parser.add_argument("--initial_dataset", type=str, default="test-data/data.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--azure_config_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="test-data/data_evol.json")
    args = parser.parse_args()

    initial_dataset = load_dataset(args.initial_dataset)

    evol_instruct = EvolInstruct(num_iterations=args.num_iterations, initial_dataset=initial_dataset,
                                 verbose=args.verbose, azure_config_path=args.azure_config_path,
                                 eliminator_model=args.eliminator_model, evol_model=args.evol_model)

    evol_instruct.evolve()

    save_dataset(args.output_path, evol_instruct.dataset)

if __name__ == '__main__':
    main()
