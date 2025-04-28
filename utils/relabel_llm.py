import csv
import json
import os
import time
from typing import List, Dict
import anthropic
import pandas as pd
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Define the category prompt
CLASSIFICATION_PROMPT = """You are an expert quiz question classifier. Your task is to categorize each question into exactly one of the defined categories below. 
For each question in the batch, respond with ONLY the category number (1-8) that best fits the question. If a question could fit multiple categories, choose the most specific or dominant category. 

CATEGORY DEFINITIONS:

1. Arts, Literature & Culture, including philosophy, mythology and religious belief & practice; plus classical music 

2. History & politics, including economic, social and religious history, past civilisations and archaeology, political systems/theory and law 

3. Geography/World, including national or ethnic customs/cultures, buildings and infrastructure; business and economics; sociology and other social sciences 

4. Entertainment: film, television & radio, and all other forms of popular entertainment; plus pop music when no Music category 

5. Sport & games, including computer games and competitive pastimes. 

6. Lifestyle: food & drink; clothes & fashion; products, brands & advertising; health & fitness; hobbies (except competitive games); celebrities; internet (websites, social media, memes etc) 

7. Music: pop & rock; classical; musicals; jazz, folk, world music; musical theory & instrument 

8. Science & nature, including maths, technology/engineering, astronomy & space exploration, medicine & anatomy, psychology"""

def read_questions_from_csv(filepath: str) -> pd.DataFrame:
    """Read questions from CSV file."""
    df = pd.read_csv(filepath)
    return df

def create_batches(df: pd.DataFrame, batch_size: int = 5) -> List[List[str]]:
    """Split questions into batches."""
    questions = df['Question'].tolist()
    return [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]

def format_prompt_with_questions(questions: List[str]) -> str:
    """Format the complete prompt with batch of questions."""
    numbered_questions = "\n\n".join([f"Question {i+1}: {q}" for i, q in enumerate(questions)])
    return f"{CLASSIFICATION_PROMPT}\n\nHere are the questions to classify:\n\n{numbered_questions}"

def classify_batch(questions: List[str]) -> List[int]:
    """Send a batch of questions to Claude API and get category classifications."""
    prompt = format_prompt_with_questions(questions)
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=100,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse the response to get category numbers (1-8)
    category_numbers = []
    response_text = response.content[0].text.strip()
    
    # Extract just the numbers from the response
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    for line in lines:
        # Try to extract the category number more robustly
        try:
            # Split line by space or '.', take the last part which should be the category
            parts = line.replace('.', ' ').split()
            if parts:
                last_part = parts[-1]
                if last_part.isdigit():
                    category_number = int(last_part)
                    if 1 <= category_number <= 8:
                        category_numbers.append(category_number)
        except Exception as e:
            print(f"Warning: Could not parse line '{line}': {e}")
            continue
    
    # Ensure we have the correct number of categories
    if len(category_numbers) != len(questions):
        print(f"Warning: Expected {len(questions)} categories but got {len(category_numbers)}")
        print(f"Response: {response_text}")
    
    return category_numbers

def map_category_number_to_name(category_number: int) -> str:
    """Map category number to its name."""
    category_map = {
        1: "Art and Literature",
        2: "History",
        3: "Geography",
        4: "Entertainment",
        5: "Sport",
        6: "Lifestyle",
        7: "Music",
        8: "Science and Nature"
    }
    return category_map.get(category_number, "Unknown")

def process_csv_file(input_filepath: str, output_filepath: str, batch_size: int = 10):
    """Process the entire CSV file and save results."""
    # Read the data
    df = read_questions_from_csv(input_filepath)
    
    # Create batches
    batches = create_batches(df, batch_size)
    
    # Initialize results list
    claude_categories = []
    
    # Process each batch
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")
        
        try:
            # Classify the batch
            categories = classify_batch(batch)
            
            # Add to results
            claude_categories.extend(categories)
            
            # Wait a bit to avoid rate limits
            if i < len(batches) - 1:
                time.sleep(2)
                
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}")
            # In case of error, append placeholder values
            claude_categories.extend([0] * len(batch))
    
    # Match the number of results to the number of questions
    # This handles edge cases if something went wrong
    if len(claude_categories) < len(df):
        claude_categories.extend([0] * (len(df) - len(claude_categories)))
    elif len(claude_categories) > len(df):
        claude_categories = claude_categories[:len(df)]
    
    # Add results to dataframe
    df['Claude_Category_Number'] = claude_categories
    df['Claude_Category'] = df['Claude_Category_Number'].apply(map_category_number_to_name)
    
    # Calculate agreement and disagreement
    df['Agreement'] = df['Claude_Category'] == df['True']
    
    # Save results
    df.to_csv(output_filepath, index=False)
    
    # Print summary
    total = len(df)
    agreements = df['Agreement'].sum()
    print(f"\nResults Summary:")
    print(f"Total questions: {total}")
    print(f"Agreements with original 'True' label: {agreements} ({agreements/total:.2%})")
    print(f"Disagreements: {total - agreements} ({(total - agreements)/total:.2%})")
    
    # Print details of disagreements by category
    print("\nDisagreements by category:")
    disagreements = df[~df['Agreement']]
    for category in df['True'].unique():
        cat_disagreements = disagreements[disagreements['True'] == category]
        if len(cat_disagreements) > 0:
            print(f"{category}: {len(cat_disagreements)} disagreements")
            
    # Print results details
    print(f"\nResults saved to {output_filepath}")

if __name__ == "__main__":
    input_file = "../reports/label_review/suspected_label_errors.csv"
    output_file = "../reports/label_review/claude_labels.csv"
    
    # Check if API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it by creating a .env file with ANTHROPIC_API_KEY=your_key or setting it in your environment")
        exit(1)
    
    process_csv_file(input_file, output_file, batch_size=20)

