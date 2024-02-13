import os
import requests
import re

# Access environment variables
owner = os.environ.get('owner')
repo = os.environ.get('repo')
github_token = os.environ.get('GITHUB_TOKEN')

# Function to get the total lines committed by a user
def get_user_lines_committed(username):
    url = f'https://api.github.com/repos/{owner}/{repo}/commits?author={username}&per_page=100'
    response = requests.get(url, headers={"Authorization": f"token {github_token}"})
    if response.status_code == 200:
        commits = response.json()
        total_lines = 0
        for commit in commits:
            sha = commit['sha']
            commit_url = f'https://api.github.com/repos/{owner}/{repo}/commits/{sha}'
            commit_response = requests.get(commit_url, headers={"Authorization": f"token {github_token}"})
            if commit_response.status_code == 200:
                changes = commit_response.json()['files']
                for change in changes:
                    total_lines += change['changes']
        return total_lines
    else:
        return None

# Function to get a list of contributors to the repository
def get_contributors():
    url = f'https://api.github.com/repos/{owner}/{repo}/contributors'
    response = requests.get(url, headers={"Authorization": f"token {github_token}"})
    if response.status_code == 200:
        contributors = response.json()
        return contributors
    else:
        return None

# Function to generate a Mermaid pie chart and update README.md
def generate_and_update_pie_chart():
    contributors = get_contributors()

    if contributors:
        data = []
        for contributor in contributors:
            username = contributor['login']
            total_lines = get_user_lines_committed(username)
            if total_lines is not None:
                data.append((username, total_lines))

        # Generate Mermaid pie chart data
        chart_data = ""
        for username, total_lines in data:
            chart_data += f'"{username}" : {total_lines}\n'

        # Update the README.md file with the pie chart data
        with open('README.md', 'r') as file:
            readme_content = file.read()

        # Use regular expressions to capture and replace the Mermaid code block
        mermaid_pattern = r'<!-- BEGIN MERMAID -->(.*?)<!-- END MERMAID -->'
        updated_readme = re.sub(mermaid_pattern, f'<!-- BEGIN MERMAID -->\n```mermaid\npie\ntitle Lines Committed by Contributors\n{chart_data}```\n<!-- END MERMAID -->', readme_content, flags=re.DOTALL)

        with open('README.md', 'w') as file:
            file.write(updated_readme)

        print("Updated README.md with the pie chart data.")
    else:
        print("Error getting contributors list")

if __name__ == '__main__':
    generate_and_update_pie_chart()
