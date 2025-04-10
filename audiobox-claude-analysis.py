import argparse
import pandas as pd
import anthropic
import os
from tqdm import tqdm
import time

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze TTS voice comparison CSV data using Claude')
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('output_md', help='Path to the output Markdown file')
    parser.add_argument('--api_key', help='Anthropic API key (optional if set as env variable)')
    parser.add_argument('--model', default='claude-3-opus-20240229', help='Claude model to use')
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided either as an argument or as ANTHROPIC_API_KEY environment variable")
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Read the CSV file
    print(f"Reading CSV file: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
        
        # Pre-process the data to calculate win rates properly
        print("Pre-processing data to calculate accurate win statistics...")
        
        # Calculate win statistics for each voice
        all_voices = set()
        for col in ['v1', 'v2', 'v3']:
            if col in df.columns:
                all_voices.update(df[col].dropna().unique())
        
        voice_stats = []
        for voice in all_voices:
            # Count appearances
            appearances = sum((df['v1'] == voice) | (df['v2'] == voice) | (df['v3'] == voice))
            # Count wins
            wins = sum(df['winner'] == voice)
            # Calculate win percentage
            win_percentage = (wins / appearances * 100) if appearances > 0 else 0
            
            voice_stats.append({
                'voice_id': voice,
                'appearances': appearances,
                'wins': wins,
                'win_percentage': round(win_percentage, 2)
            })
        
        # Sort by win percentage descending
        voice_stats_df = pd.DataFrame(voice_stats).sort_values('win_percentage', ascending=False)
        
        # Add voice stats to the original data
        voice_stats_csv = voice_stats_df.to_csv(index=False)
        
        # Convert the original data to CSV
        csv_data = df.to_csv(index=False)
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return
    
    # Create the prompt with the additional voice stats
    prompt = f"""You are tasked with analyzing a CSV file that compares automated model-based scoring and manual human scoring for text-to-speech (TTS) voices. Your goal is to identify the best voices based on both human and automated scoring, as well as to identify any relationships between human scoring/preferences and automated model scoring.

The CSV data is provided below:

<csv_data>
{csv_data}
</csv_data>

I've also calculated the win statistics for each voice, which you should use for the human preference analysis:

<voice_stats>
{voice_stats_csv}
</voice_stats>

The CSV contains the following columns:
1. id	 - A unique matchup identifier
2. text_prompt - the text prompt given to the TTS Api to generate the audio
3. v1 - voice 1 in the matchup
4. v2 - voice 2 in the matchup
5. v3 - voice 3 in the matchup
6. winner - the user preferred voice from the matchup based on the use case and text style
7. use_case -- the use case of the text that was generated
8. text_style - the text style used for hte generation of the text prompt
9. - 12  --	v1_CE	, v1_CU, v1_PC, v1_PQ -- automated voice scores from audiobox-aesthetics for v1 voice 1 in the matchup. The definitions are listed below. 
13 - 16 -- v2_CE,	v2_CU, v2_PC, v2_PQ -- 	automated voice scores from audiobox-aesthetics for v2 voice 2 in the matchup. The definitions are listed below. 
17 - 20 -- v3_CE, v3_CU, v3_PC, v3_PQ	-- automated voice scores from audiobox-aesthetics for v3 voice 3 in the matchup. The definitions are listed below. 


CE	Content Enjoyment 
CU	Content Usefulness
PC	Production Complexity
PQ	Production Quality		

1. Production Quality (PQ) Focuses on the technical aspects of quality instead of subjective quality. Aspects
including clarity & fidelity, dynamics, frequencies and spatialization of the audio;
2. Production Complexity (PC) Focuses on the complexity of an audio scene, measured by number of audio
components;
3. Content Enjoyment (CE). Focuses on the subject quality of an audio piece. It's a more open-ended axis,
some aspects might includes emotional impact, artistic skill, artistic expression, as well as subjective
experience, etc;
4. Content Usefulness (CU) Also a subjective axis, evaluating the likelihood of leveraging the audio as
source material for content creation.					

To analyze this data, please follow these steps:

1. Automated model scoring analysis:
   a. Calculate the average scores for CE, CU, PC, and PQ across all voices.
   b. Identify the top 5 voices for each automated scoring metric.
   c. Determine if there are any voices that consistently perform well across all automated metrics.

2. Human scoring analysis:
   a. Use the provided voice_stats table to identify top performing voices by win percentage
   b. Make sure to only consider voices with a significant number of appearances (at least 30)
   c. Report the top 5 voices by win percentage and their exact win rates as shown in the voice_stats table

3. Relationship between automated and human scoring:
   a. Calculate the correlation coefficients between each automated scoring metric (CE, CU, PC, PQ) and each human scoring metric (win percentage).
   b. Identify any strong positive or negative correlations between automated and human scoring metrics.

4. Identify the best voices:
   a. Create a composite score that combines both automated and human scoring metrics.
   b. Rank the voices based on this composite score.
   c. Identify the top 5 overall voices.
   d. Identify the top voices based on use case and text_style categories. 

5. Additional insights:
   a. Analyze if there are any patterns or trends in the data that might be interesting or unexpected.
   b. Consider if certain voices perform better for specific types of text or content.

Present your findings in the following format:

<analysis>
1. Automated Scoring Results:
   [Include top 5 voices for each metric and any consistently high-performing voices]

2. Human Scoring Results:
   [Include top 5 voices by win percentage using the provided voice_stats data]

3. Relationship Between Automated and Human Scoring:
   [Describe any strong correlations or interesting patterns]

4. Best Overall Voices:
   [List the top 5 voices based on the composite score]

5. Additional Insights:
   [Describe any other interesting findings or patterns in the data]

6. Conclusions and Recommendations:
   [Summarize the key findings and provide recommendations for voice selection or further analysis]
</analysis>

Remember to focus on the most important findings and insights in your analysis. Your final output should only include the content within the <analysis> tags, without any additional commentary or explanations outside of these tags."""

    # Call the Anthropic API with progress tracking
    print("Sending request to Claude API...")
    
    with tqdm(total=100, desc="Analyzing data") as pbar:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            temperature=1,
            system="You are a data analyst specializing in voice technology evaluation.",
            messages=[
                {"role": "user", "content": prompt}
            ],
            thinking={
                "type": "enabled",
                "budget_tokens": 16000
            }
        )
        
        # Simulate progress since we don't get actual progress from the API
        for _ in range(10): # 10 is the number of tokens in the prompt  
            time.sleep(0.5)
            pbar.update(10)
    
    # Extract just the analysis portion
    response = message.content[0].text
    analysis_start = response.find("<analysis>")
    analysis_end = response.find("</analysis>") + len("</analysis>")
    
    if analysis_start != -1 and analysis_end != -1:
        analysis = response[analysis_start:analysis_end]
    else:
        analysis = response  # Fallback to full response if tags not found
    
    # Write to output file
    print(f"Writing analysis to: {args.output_md}")
    with open(args.output_md, 'w', encoding='utf-8') as f:
        f.write(analysis)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()