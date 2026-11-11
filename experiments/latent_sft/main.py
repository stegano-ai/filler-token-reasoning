from research_scaffold.config_tools import execute_experiments
from research_scaffold.argparsing import get_base_argparser, process_base_args
from functions import train_curriculum, train_latent_sft

function_map = {
    "train_latent_sft": train_latent_sft,
    "train_curriculum": train_curriculum,
}

if __name__ == "__main__":
    parser = get_base_argparser()
    args = parser.parse_args()
    config_path, meta_config_path, sweep_config_path = process_base_args(args)
    execute_experiments(
        function_map=function_map,
        config_path=config_path,
        meta_config_path=meta_config_path,
        sweep_config_path=sweep_config_path,
        dry_run=args.dry_run,
    )
