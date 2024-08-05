import re

# Read the content of the file
with open('reinforcementagent.py', 'r') as file:
    content = file.read()

# Define the regex pattern
pattern = r'^\s*# In\[\d+\]:\s*\n\s*\n?'

# Replace the matched patterns with an empty string
cleaned_content = re.sub(pattern, '', content, flags=re.MULTILINE)

# Write the cleaned content back to the file
with open('reinforcementagent.py', 'w') as file:
    file.write(cleaned_content)