import utils
import logging

if __name__=="__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    utils.clean_hf_dataset_caches()






