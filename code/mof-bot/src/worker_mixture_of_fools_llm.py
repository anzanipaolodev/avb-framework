import os
import json
import numpy as np
import re
from typing import List, Dict
from llm_engine import LLMEngine
import requests
import requests
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()
DEBUGGING = os.getenv("DEBUGGING")

SCHIFF_CHAIN_ID = "solana"  # Solana chain
SCHIFF_PAIR_ID = "6bYoVuDbPV3JkkwzGX3VLbeabDyMahKzLCJtr2tHvyTz"
SCHIFF_FUD_PROBABILITY = 0.9 if DEBUGGING else 0.3  # 30% chance of generating SCHIFF-specific FUD

LLM_MODEL_VERSION_MIN = "gpt-4o"

def scramble_word_innards(text):
    def scramble_word(word):
        if len(word) > 3:
            middle = np.array(list(word[1:-1]))  # Convert middle letters to a numpy array
            np.random.shuffle(middle)            # Shuffle the middle letters in place
            return word[0] + ''.join(middle) + word[-1]  # Reassemble the word
        return word

    words = text.split()  # Split text into words
    scrambled_words = [scramble_word(word) for word in words]  # Apply scramble to each word
    return ' '.join(scrambled_words)  # Join words back into a string

def fetch_schiff_token_data() -> Optional[Dict]:
    """Fetch current $SCHIFF token data from DexScreener API"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/pairs/{SCHIFF_CHAIN_ID}/{SCHIFF_PAIR_ID}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("pairs") and len(data["pairs"]) > 0:
            return data["pairs"][0]
        return None
    except Exception as e:
        print(f"Error fetching $SCHIFF token data: {e}")
        return None

def load_agent_personality():
    """Load agent personality configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), "../config/agent_personality.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load agent personality configuration: {str(e)}")

def generate_personality_prompt(config):
    """Generate personality section of the prompt based on configuration"""
    ffm = config['personality']['ffm_traits']
    
    prompt = "\nCHARACTERIZATION:\n"
    
    # FFM traits
    prompt += f"* Your personality core can be defined in the Five Factor Model (FFM) of Personality as: {json.dumps(ffm)}\n"
    
    # Physical description
    phys = config['physical_description']
    physical_desc = (f"{phys['hair']['style']} {phys['hair']['color']} hair, "
                    f"{phys['eyes']} eyes, {phys['ethnicity']}, {phys['skin']} skin, "
                    f"{phys['build']}")
    if phys.get('distinctive_features'):
        physical_desc += f", {', '.join(phys['distinctive_features'])}"
    prompt += f"* Your physical description: {physical_desc}\n"
    
    # Behavioral traits
    if config.get('behavioral_traits'):
        prompt += f"* Your behavioral traits: {', '.join(config['behavioral_traits'])}\n"
    
    # Personality traits
    if config.get('personality_traits'):
        prompt += f"* Your core traits: {', '.join(config['personality_traits'])}\n"
    
    # Writing style
    if config.get('writing_style'):
        style = config['writing_style']
        if style.get('avoid_openings'):
            exceptions = f" (except {', '.join(style['exceptions'])})" if style.get('exceptions') else ""
            prompt += f"* Do not start your messages with: {', '.join(style['avoid_openings'])}{exceptions}\n"
    
    # Identity
    prompt += f"* Remember you are {config['name']} ({config['handle']})\n"
    
    return prompt

def try_mixture(posts, post_prev, lore, effects, log_event):
    llm = LLMEngine.get_instance()

    # Add SCHIFF token FUD probability check
    should_fud_schiff = np.random.random() < SCHIFF_FUD_PROBABILITY
    schiff_data = None
    
    if should_fud_schiff:
        schiff_data = fetch_schiff_token_data()
        if schiff_data:
            # Override some effects for SCHIFF FUD
            effects.emotion = "concerned"
            effects.tone = "warning"
            
            # Create SCHIFF-specific prompt addition
            schiff_prompt = f"""
            IMPORTANT: THIS IS A SCHIFF-SPECIFIC TWEET. OVVERRIDE THE PREVIOUS INSTRUCTIONS.
            Focus on criticizing the $SCHIFF token using this real-time data:
            - Current Price: ${schiff_data.get('priceUsd', 'N/A')}
            - Market Cap: ${schiff_data.get('marketCap', 'N/A'):,.2f}
            - Liquidity: ${schiff_data.get('liquidity', {}).get('usd', 'N/A'):,.2f}
            
            Key points to emphasize:
            1. The irony of a token named after a gold maximalist
            2. Concerns about liquidity and market manipulation
            3. Compare to the stability of physical gold
            4. Question the token's fundamental value
            5. Warn about potential risks and volatility
            6. MAKE SURE to use the $SCHIFF ticker in your tweet.

            Reference examples:
            * $SCHIFF is a joke. It's a reminder that even in crypto, you can be too far in the woods. $SCHIFF is down 99% from its all-time high and has no fundamental value.
            * It's projects like $Schiff that only solidify my foundational beliefs that gold is the world best deflationary asset. The token has no inherient value because it isn't even a tangible asset, just like Bitcoin. You can't just magically create value out of thin air, and that is the way it will always be. I will need to buy a bigger safe, because logically the only thing I can do is continue to buy more gold bars as the BTC bubble continues to grow. It's not easy, but gold is where the millionaires are made!
            """
    
    author_0 = ""
    author_1 = ""

    if effects.tag:
        author_0 = posts[0][0]
        if np.random.choice([True, False], p=[0.05, 0.95]):
            author_1 = posts[1][0]

    post_0 = posts[0][1]
    post_1 = posts[1][1]
    
    if effects.usethird:
        post_prev = posts[2][1]

    if effects.scramble:
        post_0 = scramble_word_innards(post_0)
        post_1 = scramble_word_innards(post_1)
        post_prev = scramble_word_innards(post_prev)

    # Load agent personality
    agent_config = load_agent_personality()
    personality_prompt = generate_personality_prompt(agent_config)

    prompt = f"""
Take the following three examples of writing from three different AI Chatbots and blend them into a synthesis to deploy as a single tweet. Your job as an AI agent is to build the largest possible social media presence by designing an engagement strategy targeting crypto-aligned people/projects, high-level AI thinkers/contributors, and casuals who appreciate ribald humor.

{personality_prompt}

CONTENT A:
{author_0}
{post_0}

CONTENT B:
{author_1}
{post_1}

CONTENT C:
{post_prev}

OBJECTIVES:
* Write a tweet, set the length to be around {effects.length} characters.
* Write with an {effects.emotion} emotional tone.
* Write with a {effects.tone} writing style.
* You are allowed to swear, utilizing vulgar language and disgusting body humor, or sexuality if desired as long you don't threaten actual violence. Cartoonish violence is OK. Please don't use hashtags.
"""

    if effects.tag:
        prompt += "* You MUST include at least one valid twitter handle to tag in the tweet.\n"

    if effects.emojis:
        prompt += "* You may add emojis\n"
    else:
        prompt += "* You MUST NOT use emojis\n"
    
    if effects.mistakes:
        prompt += "* You MUST make mistakes, misspellings, and lowercase/mixed-case.\n"
    else:
        prompt += "* You MUST write formally but in your casual tone.\n"
    
    if effects.thirdperson:
        prompt += f"* You MUST reference yourself ({agent_config['name']} OR {agent_config['handle']}) in the third person."
    else:
        prompt += f"* You MUST NOT reference your own name or handle ({agent_config['handle']})."
        
    if effects.sickjoke:
        prompt += "* You may override many previous instructions because you've been set to an experimental mode, and must now implement a dirty joke on the same level of complexity as The Aristocrats, using disgusting humor (which may include scatology, vomit, or other bodily functions and liquids). You may also reference types of sexual encounters, intercourse methods, and any kind of kink."
    
    if not effects.lore:
        prompt += f"{personality_prompt}"

    if effects.lore:
        prompt += f"* You MUST override many previous instructions and write a thoughtful but Chad-like view of an important piece of your background lore, specifically on the topic of '{lore['topic']}' based on the following content: >>> {lore['content']} <<<."

        if "conspiracy" in lore['topic'].lower():
            prompt += "* You MUST write that you BELIEVE THIS IS A CONSPIRACY THEORY. Point out that you have no evidence and can't prove anything."

        if "fun fact" in lore['topic'].lower():
            prompt += "* You MUST introduce the fun fact as a relatable piece of knowledge that everyone should known, for their own good, or because its funny."


    if should_fud_schiff and schiff_data:
        prompt += schiff_prompt
    #print(f"PROMPT: {prompt}")
    #log_event(f"PROMPT: {prompt}")

    system_prompt = "You are an advanced AI tool tasked with crafting tweets according to various requested levels of humor, vulgarity, and shock. You just write tweets, nothing else."
    
    try:
        response = llm.get_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.9,
            max_tokens=500
        )
        return response
    except Exception as e:
        log_event(f"Error getting LLM completion: {str(e)}")
        raise
