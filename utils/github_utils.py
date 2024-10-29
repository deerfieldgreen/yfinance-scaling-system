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
    Pushes files from the data directory to a GitHub repository, creating a new branch
    and pull request if there are any changes.

    Args:
        repo_name (str): The name of the GitHub repository (e.g., "user/repo").
        file_dir (str): The directory within the repository where the files should be stored.
        token (str): Your GitHub personal access token.
    """

    repo_name = "deerfieldgreen/yfinance-scaling-system"  # Replace with your repo details
    file_dir = "data"
    token = os.environ.get("GIT_TOKEN")  # Get token from environment variable
    data_path = "data/"  # Path to your data directory
    branch_name = f"auto-update-{datetime.datetime.now(pytz.utc).strftime('%Y%m%d%H%M%S')}"

    # Get the current time in UTC
    current_time = datetime.datetime.now(pytz.utc)

    # Prepare GitHub
    g = Github(token)
    repo = g.get_repo(repo_name)

    # Create a new branch
    source = repo.get_branch("main")
    try:
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=source.commit.sha)
        logger.info(f"Created branch: {branch_name}")
    except GithubException as e:
        if e.status == 422:  # Branch already exists
            logger.warning(f"Branch {branch_name} already exists.")
        else:
            raise e

    files_to_push = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path):
            # Get last modified time of the file
            modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path), pytz.utc)

            # Calculate time difference in hours
            time_diff = current_time - modified_time
            hours_diff = time_diff.total_seconds() / 3600

            if hours_diff <= 1:  # Check if modified in the last hour
                logger.info(f"File {filename} has been modified in the last hour. Adding to push list...")
                files_to_push.append(filename)

    if files_to_push:
        for filename in files_to_push:
            file_path = os.path.join(data_path, filename)
            # Read the file content
            content = read_and_encode_file(file_path, encode=False)

            try:
                git_file = repo.get_contents(f"{file_dir}/{filename}", ref=branch_name)
                logger.info(f"Found existing file: {git_file.path}")
                repo.update_file(
                    git_file.path,
                    f"Updated file for {current_time.date()}",
                    content,
                    git_file.sha,
                    branch=branch_name
                )
            except GithubException as e:
                if e.status == 404:  # File not found
                    logger.info(f"File not found. Creating a new file.")
                    repo.create_file(
                        f"{file_dir}/{filename}",
                        f"Created file for {current_time.date()}",
                        content,
                        branch=branch_name
                    )
                else:
                    logger.error(f"An error occurred: {str(e)}")
                    raise e
        
        # Create a pull request
        try:
            pr = repo.create_pull(
                title=f"Automated data update for {current_time.date()}",
                body="This pull request contains automated data updates.",
                head=branch_name,
                base="main"
            )
            logger.info(f"Pull request created: {pr.html_url}")
        except GithubException as e:
            logger.error(f"An error occurred while creating the pull request: {str(e)}")
            raise e

    else:
        logger.info(f"No files modified in the last hour. Skipping push to GitHub.")

    logger.info(f"Push to Github complete.")

if __name__ == "__main__":
    push_to_github()


