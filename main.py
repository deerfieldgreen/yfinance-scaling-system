# main.py

import logging
from utils.stock_splits import generate_stock_splits
from utils.github_utils import push_to_github

def main():

  # exit(0) # temporary to stop endless loop

  ## All functions push their data to the ../data/ directory
  generate_stock_splits()
  ## Add other modules here


  ## Then all files in data get committ-pushed
  ## Push to GitHub comes last
  push_to_github()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting my Cloud Run Job")

    try:
        main()
        logging.info("Job completed successfully")
    except Exception as e:
        logging.error(f"Job failed: {e}")
