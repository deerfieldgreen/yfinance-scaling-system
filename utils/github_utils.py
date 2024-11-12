import os
import logging
from github import Github, GithubException

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
    Pushes files from the data directory directly to the main branch of a GitHub repository.

    Args:
        repo_name (str): The name of the GitHub repository (e.g., "user/repo").
        file_dir (str): The directory within the repository where the files should be stored.
        token (str): Your GitHub personal access token.
    """

    repo_name = "deerfieldgreen/yfinance-scaling-system"  # Replace with your repo details
    file_dir = "data"
    token = os.environ.get("GIT_TOKEN")  # Get token from environment variable
    data_path = "data/"  # Path to your data directory

    # Prepare GitHub
    g = Github(token)
    repo = g.get_repo(repo_name)

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path):
            # Read the file content
            content = read_and_encode_file(file_path, encode=False) 

            try:
                git_file = repo.get_contents(f"{file_dir}/{filename}", ref="main")
                logger.info(f"Found existing file: {git_file.path}")
                # Update the file
                repo.update_file(
                    git_file.path,
                    f"Updated file", 
                    content,
                    git_file.sha,
                    branch="main"
                )
            except GithubException as e:
                if e.status == 404:  # File not found
                    logger.info(f"File not found. Creating a new file.")
                    # Create the file
                    repo.create_file(
                        f"{file_dir}/{filename}",
                        f"Created file", 
                        content,
                        branch="main"
                    )
                else:
                    logger.error(f"An error occurred: {str(e)}")
                    raise e

    logger.info(f"Push to Github complete.")

if __name__ == "__main__":
    push_to_github()