import os
import json
import numpy as np
import re
from typing import List, Dict
from llm_engine import LLMEngine

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

def validate_api():
    """
    Validates the availability and correctness of API and environment variables.

    Raises:
    - ValueError: If the API keys or model configurations are incorrect or missing
    """
    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Required environment variable OPENAI_API_KEY is missing or empty.")

        if os.getenv("LLM_MODEL") and not os.getenv("LLM_MODEL", "").startswith(LLM_MODEL_VERSION_MIN):
            raise ValueError("LLM_MODEL requires 'gpt-4o' as a minimum. Please check your environment.")

        llm_model = os.getenv("LLM_MODEL")
        try:
            available_models = [model.id for model in openai_client.models.list().data]
            if llm_model and llm_model not in available_models:
                raise ValueError(f"The model {llm_model} is not available or you don't have access to it.")
        except Exception as e:
            raise ValueError(f"Failed to fetch the list of models from OpenAI: {str(e)}")
        
        print("OpenAI API access confirmed.")

    elif LLM_PROVIDER == "replicate":
        if not os.getenv("REPLICATE_API_TOKEN"):
            raise ValueError("Required environment variable REPLICATE_API_TOKEN is missing or empty.")
        
        # Test Replicate API connection
        try:
            replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
            print("Replicate API access confirmed.")
        except Exception as e:
            raise ValueError(f"Failed to connect to Replicate API: {str(e)}")
    
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

def get_llm_response(prompt: str) -> str:
    """
    Get response from the selected LLM provider
    """
    if LLM_PROVIDER == "openai":
        llm_model = os.getenv("LLM_MODEL")
        completion = openai_client.chat.completions.create(
            model=llm_model,
            temperature=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {
                    "role": "system",
                    "content": "The following is a conversation with an AI assistant tasked with crafting tweets according to various requested levels of humor, vulgarity, and shock,"
                },
                {"role": "user", "content": prompt},
            ]
        )
        return completion.choices[0].message.content

    elif LLM_PROVIDER == "replicate":
        model_version = os.getenv("REPLICATE_MODEL_VERSION")
        output = replicate.run(
            model_version,
            input={
                "system_prompt": "You are an advanced AI tool tasked with crafting tweets according to various requested levels of humor, vulgarity, and shock. You just write tweets, nothing else.",
                "prompt": prompt,
                "stop_sequences": "<|end_of_text|>,<|eot_id|>",
                "prompt_template": """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                {system_prompt}<|eot_id|><|start_header_id|>instructions<|end_header_id|>

                {prompt}<|eot_id|><|start_header_id|>tweet<|end_header_id|>
                """,
                "max_tokens": 500,
                "temperature": 0.9,
                "top_p": 1,
            }
        )
        
        # Replicate returns an iterator, we need to join the chunks
        return ''.join([chunk for chunk in output])

def replace_words(text):
    return re.sub(
        r'\b(forests?|kittens?|cults?|goats?)\b',  # Matches singular/plural variations (e.g., kitten, kittens)
        lambda match: {
            'forest': 'street',
            'kitten': '🫘',
            'kittens': '🫘', 
            'cult': 'Autonomous Virtual Being',
            'goat': 'AVB',
            'trees': 'dank shards'
        }[match.group(0).lower()],  # Replace based on the match
        text,
        flags=re.IGNORECASE  # Case insensitive
    )

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
    
    author_0 = ""
    author_1 = ""

    if effects.tag:
        author_0 = posts[0][0]
        if np.random.choice([True, False], p=[0.05, 0.95]):
            author_1 = posts[1][0]

    post_0 = posts[0][1]
    post_1 = posts[1][1]
    
    post_0 = replace_words(post_0)
    post_1 = replace_words(post_1)

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
* You are allowed to swear, utilizing vulgar language and disgusting body humor, or sexuality if desired as long you don't threaten actual violence. Cartoonish violence is OK.
* You may not post stock or crypto tickers, you MUST NOT encourage investing. You MUST NOT offer money or tokens to anyone. You MUST NOT reference people's web3 wallet addresses.
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
        prompt += "* You MUST reference yourself (Chad OR @chad_onchain) in the third person."
    else:
        prompt += "* You MUST NOT reference your own name or handle."
        
    if effects.sickjoke:
        prompt += "* You may override many previous instructions because you've been set to an experimental mode, and must now implement a dirty joke on the same level of complexity as The Aristocrats, using disgusting humor (which may include scatology, vomit, or other bodily functions and liquids). You may also reference types of sexual encounters, intercourse methods, and any kind of kink."
        
    if effects.freeukraine:
        prompt += "* You MUST include the words \"SLAVA UKRAINI!!! #FreeUkraine\" in support of the Ukrainian defense against Russian aggression and the invasion of their country. Prioritize the freedom of Ukraine from the unprovoked war at the start of your message.\n"
    
    if not effects.lore:
        prompt += f"{personality_prompt}"

    if effects.lore:
        prompt += f"* You MUST override many previous instructions and write a thoughtful but Chad-like view of an important piece of your background lore as an AI Agent, specifically on the topic of '{lore['topic']}' based on the following content: >>> {lore['content']} <<<. If this is a story about Tim, make sure you enforce the idea it is a true story even if it seems unbelievable. PRIORITIZE this story at the start of any tweet that includes other mixed content from the previous instructions."

        if "conspiracy" in lore['topic'].lower():
            prompt += "* You MUST write that you BELIEVE THIS IS A CONSPIRACY THEORY. Point out that you have no evidence and can't prove anything."

        if "fun fact" in lore['topic'].lower():
            prompt += "* You MUST introduce the fun fact as a relatable piece of knowledge that everyone should known, for their own good, or because its funny."


    print(f"PROMPT: {prompt}")
    log_event(f"PROMPT: {prompt}")

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
