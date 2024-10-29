import pandas as pd
import os
from github import Github, GithubException
import base64
import datetime
import pytz
import dotenv
from stocksplits import splits

dotenv.load_dotenv()

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_and_encode_file(file_path, encode=True):
    """Read the file content and encode it in base64."""
    with open(file_path, 'rb') as file:
        content = file.read()
    if encode:
        return base64.b64encode(content).decode('utf-8')
    else:
        return content


if __name__ == '__main__':

    # Define variables for paths and filenames
    work_dir = os.getcwd()
    repo_name = "deerfieldgreen/yfinance-scaling-system"  # Replace with your repo details
    file_dir = "stock-splits"
    file_name = "stock_splits_data.csv"
    csv_path = os.path.join(work_dir, file_dir, file_name)

    # Get the current time in UTC
    current_time = datetime.datetime.now(pytz.utc)


    # Read the stock symbols from the text file
  with open("symbols.txt", "r") as f:
      symbols = [line.strip() for line in f]

  # Get today's date
  today = datetime.date.today()

  # Calculate the date 60 months ago
  lookback_months = today - datetime.timedelta(days=60 * 30.5)
  lookback_months = pd.Timestamp(lookback_months, tz='America/New_York')

  # Fetch and save the stock split data
  df = get_stock_splits(symbols, lookback_months)
  df.to_csv("stock_splits_data.csv", index=False)
    # Generate the dataframe using the `splits` function
    df = splits.get_splits_data()  # Call the function within the module




    # Save the dataframe to a CSV file
    df.to_csv(csv_path, index=False)

    # Prepare github
    token = os.environ.get("GIT_TOKEN")
    g = Github(token)
    repo = g.get_repo(repo_name)

    # Read the CSV file content
    content = read_and_encode_file(csv_path, encode=False)

    try:
        git_file = repo.get_contents(f"{file_dir}/{file_name}")
        logger.info(f"Found existing file: {git_file.path}")
        repo.update_file(
            git_file.path,
            f"Updated file for {current_time.date()}",
            content,
            git_file.sha,
        )
    except Exception as e:
        if isinstance(e, GithubException) and e.status == 404:  # File not found
            logger.info(f"File not found. Creating a new file.")
            repo.create_file(
                f"{file_dir}/{file_name}",
                f"Created file for {current_time.date()}",
                content,
            )
        else:
            logger.error(f"An error occurred: {str(e)}")
            raise e

    logger.info(f"Pushed to Github")


