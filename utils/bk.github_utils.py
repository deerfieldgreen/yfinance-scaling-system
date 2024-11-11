import os
import logging
from github import Github, GithubException
import base64
import datetime
import pytz

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

def push_to_github():
    """
    Pushes files from the data directory directly to the main branch of a GitHub repository
    if there are any changes within the last 15 minutes.

    Args:
        repo_name (str): The name of the GitHub repository (e.g., "user/repo").
        file_dir (str): The directory within the repository where the files should be stored.
        token (str): Your GitHub personal access token.
    """

    repo_name = "deerfieldgreen/yfinance-scaling-system"  # Replace with your repo details
    file_dir = "data"
    token = os.environ.get("GIT_TOKEN")  # Get token from environment variable
    data_path = "data/"  # Path to your data directory

    # Get the current time in UTC
    current_time = datetime.datetime.now(pytz.utc)

    # Prepare GitHub
    g = Github(token)
    repo = g.get_repo(repo_name)

    files_to_push = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path):
            # Get last modified time of the file
            modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path), pytz.utc)

            # Calculate time difference in hours
            time_diff = current_time - modified_time
            hours_diff = time_diff.total_seconds() / 900

            if hours_diff <= 1:  # Check if modified in the last hour
                logger.info(f"File {filename} has been modified in the 15 minutes hour. Adding to push list...")
                files_to_push.append(filename)

    if files_to_push:
        for filename in files_to_push:
            file_path = os.path.join(data_path, filename)
            # Read the file content
            content = read_and_encode_file(file_path, encode=False)

            try:
                git_file = repo.get_contents(f"{file_dir}/{filename}", ref="main")
                logger.info(f"Found existing file: {git_file.path}")

                # Check the last modified time of the file on GitHub
                git_file_last_modified = git_file.last_modified
                git_file_modified_time = datetime.datetime.strptime(git_file_last_modified, "%a, %d %b %Y %H:%M:%S %Z")
                git_file_modified_time = git_file_modified_time.replace(tzinfo=pytz.utc)

                # Calculate time difference in hours
                git_time_diff = current_time - git_file_modified_time
                git_hours_diff = git_time_diff.total_seconds() / 900

                if git_hours_diff > 1:  # Only update if the file on GitHub hasn't been modified in the last hour
                    repo.update_file(
                        git_file.path,
                        f"Updated file for {current_time.date()}",
                        content,
                        git_file.sha,
                        branch="main"
                    )
                else:
                    logger.info(f"File {filename} on GitHub has been modified in the last 15 minutes. Skipping update.")

            except GithubException as e:
                if e.status == 404:  # File not found
                    logger.info(f"File not found. Creating a new file.")
                    repo.create_file(
                        f"{file_dir}/{filename}",
                        f"Created file for {current_time.date()}",
                        content,
                        branch="main"
                    )
                else:
                    logger.error(f"An error occurred: {str(e)}")
                    raise e

    else:
        logger.info(f"No files modified in the last hour. Skipping push to GitHub.")

    logger.info(f"Push to Github complete.")

if __name__ == "__main__":
    push_to_github()
