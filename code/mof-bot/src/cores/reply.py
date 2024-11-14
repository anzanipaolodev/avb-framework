from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict
import tweepy
from tweepy.errors import TweepyException, TooManyRequests
import asyncio
import os
import json
from dotenv import load_dotenv

from cores.avbcore import AVBCore
from logger import EventLogger
from llm_engine import LLMEngine
from worker_pick_random_effects import pick_effects

load_dotenv()
DEBUGGING = os.getenv("DEBUGGING")

@dataclass
class DebuggingTweet:
    """Class for debugging tweets"""
    id: str
    text: str

@dataclass
class DebuggingTweets:
    """Class for debugging tweets"""
    data: List[DebuggingTweet]

@dataclass
class ReplyEvent:
    target_account: str
    tweet_id: str
    content: str
    scheduled_time: datetime
    original_tweet_text: str
    completed: bool = False
    last_error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    backoff_time: int = 0  # Minutes to wait after a failed attempt

    def apply_backoff(self):
        """Apply exponential backoff after failures."""
        if self.backoff_time == 0:
            self.backoff_time = 5  # Start with 5 minutes
        else:
            self.backoff_time *= 2  # Double the backoff time
        self.scheduled_time = datetime.now() + timedelta(minutes=self.backoff_time)
        self.retries += 1

class ReplyCore(AVBCore):
    def __init__(self):
        super().__init__("Reply")
        self.target_accounts: List[str] = []
        self.last_tweets: Dict[str, str] = {}  # Store last tweet ID for each account
        self.scheduled_replies: List[ReplyEvent] = []
        self.check_interval = 15 if DEBUGGING else 5400  # 1.5 hours in seconds
        self.reply_interval = 10 if DEBUGGING else 600   # 10 minutes between replies
        self.last_check_time = None
        self.twitter_client = None
        self.logger = EventLogger(None, None)  # Get singleton instance
        self.llm = LLMEngine.get_instance()  # Get LLM instance
        
    def initialize(self):
        """Initialize the reply core."""
        try:
            self.logger.async_log(f"Initializing {self.core_name} core...")
            self._initialize_twitter_client()
            self.load_target_accounts()
            self.last_check_time = datetime.now()
            self.activate()
        except Exception as e:
            self.logger.async_log(f"Error initializing Reply core: {e}", color="red")
            raise

    def _initialize_twitter_client(self):
        """Initialize Twitter API client."""
        load_dotenv()
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        consumer_key = os.getenv("TWITTER_API_KEY")
        consumer_secret = os.getenv("TWITTER_API_SECRET")

        if not all([access_token, access_token_secret, consumer_key, consumer_secret]):
            raise ValueError("Missing Twitter API credentials")

        self.twitter_client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
        
    def load_target_accounts(self):
        """Load target accounts from configuration file."""
        config_path = os.path.join(os.path.dirname(__file__), "../../config/target_accounts.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Extract just the handles from the configuration
                self.target_accounts = [account["handle"] for account in config["target_accounts"]]
                self.logger.async_log(f"Loaded {len(self.target_accounts)} target accounts")
        except FileNotFoundError:
            self.logger.async_log("Target accounts configuration file not found, using defaults", color="yellow")
            self.target_accounts = ["PeterSchiff"]  # Fallback to default
        except json.JSONDecodeError as e:
            self.logger.async_log(f"Error parsing target accounts configuration: {e}", color="red")
            self.target_accounts = ["PeterSchiff"]  # Fallback to default
        except Exception as e:
            self.logger.async_log(f"Unexpected error loading target accounts: {e}", color="red")
            self.target_accounts = ["PeterSchiff"]
        self.logger.async_log(f"Loaded {len(self.target_accounts)} target accounts")
        
    def _tick(self):
        """Handle periodic checks and schedule replies."""
        try:
            now = datetime.now()
            
            # Check for new tweets every 15 minutes
            if (self.last_check_time is None or 
                (now - self.last_check_time).total_seconds() >= self.check_interval):
                self._check_new_tweets()
                self.last_check_time = now
                
            # Process scheduled replies
            self._process_scheduled_replies(now)
            
        except Exception as e:
            self.logger.async_log(f"Error in Reply core tick: {e}", color="red")

    def _check_new_tweets(self):
        """Check for new tweets from target accounts."""
        self.logger.async_log(f"Checking for new tweets from {len(self.target_accounts)} accounts")
        for account in self.target_accounts:
            try:
                # Get user's tweets
                if DEBUGGING:
                    # Load test data file
                    test_data_path = os.path.join(os.path.dirname(__file__), "../../data/reply_testingdata.json")
                    try:
                        with open(test_data_path, 'r', encoding='utf-8') as f:
                            test_data = json.load(f)
            
                        # Get the test tweet for the current account
                        if account in test_data["test_tweets"]:
                            test_tweet = test_data["test_tweets"][account]
                            tweets = DebuggingTweets(
                                data=[
                                    DebuggingTweet(
                                        id=test_tweet["id"],
                                        text=test_tweet["text"]
                                    ),
                                ]
                            )
                        else:
                            self.logger.async_log(f"No test tweet found for account {account}", color="yellow")
                            continue
            
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        self.logger.async_log(f"Error loading test data: {e}", color="red")
                        continue
                else:
                    user = self.twitter_client.get_user(username=account)
                    if not user.data:
                        continue
                        
                    tweets = self.twitter_client.get_users_tweets(
                        user.data.id,
                        max_results=5,
                        exclude=['retweets', 'replies']
                    )
                
                if not tweets.data:
                    continue
                    
                latest_tweet = tweets.data[0]
                
                # Check if this is a new tweet
                if (account not in self.last_tweets or 
                    latest_tweet.id != self.last_tweets[account]):
                    self._schedule_reply(account, latest_tweet)
                    self.last_tweets[account] = latest_tweet.id
                    
            except TooManyRequests:
                self.logger.async_log(f"Rate limit hit checking {account}", color="yellow")
                break
            except Exception as e:
                self.logger.async_log(f"Error checking {account}: {e}", color="red")

    def _schedule_reply(self, account: str, tweet):
        """Schedule a new reply with appropriate timing."""
        # Calculate next available slot
        if self.scheduled_replies:
            last_scheduled = max(r.scheduled_time for r in self.scheduled_replies)
            schedule_time = last_scheduled + timedelta(seconds=self.reply_interval)
        else:
            schedule_time = datetime.now() + timedelta(seconds=60)

        content = self._generate_reply_content(tweet.text, account)
        
        reply_event = ReplyEvent(
            target_account=account,
            tweet_id=tweet.id,
            content=content,
            scheduled_time=schedule_time,
            original_tweet_text=tweet.text
        )
        
        self.scheduled_replies.append(reply_event)
        self.logger.async_log(
            f"Scheduled reply to {account} at {schedule_time}"
        )

    def _process_scheduled_replies(self, current_time: datetime):
        """Process any ready replies."""
        for event in self.scheduled_replies[:]:  # Copy list to allow modification
            if not event.completed and current_time >= event.scheduled_time:
                try:
                    if not DEBUGGING:
                        self._send_reply(event)
                    event.completed = True
                    self.scheduled_replies.remove(event)
                    self.logger.async_log(f"Successfully replied to {event.target_account}")
                except TooManyRequests:
                    event.apply_backoff()
                    self.logger.async_log(
                            f"Rate limit hit, retrying in {event.backoff_time} minutes",
                        color="yellow"
                    )
                except Exception as e:
                    event.last_error = str(e)
                    self.logger.async_log(f"Reply error: {e}", color="red")
                if event.retries >= event.max_retries:
                    self.scheduled_replies.remove(event)

    def _load_personality(self) -> Dict:
        """Load the agent's personality configuration"""
        config_path = os.path.join(os.path.dirname(__file__), "../../config/agent_personality.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.async_log(f"Error loading personality config: {e}", color="red")
            return {}

    def _generate_reply_content(self, original_tweet: str, account: str) -> str:
        """Generate reply content using LLM and personality configuration"""
        self.logger.async_log(f"Generating reply content for {account}, original tweet: {original_tweet}")
        try:
            personality = self._load_personality()
            if not personality:
                return None

            # Extract relevant personality traits
            writing_style = personality.get('writing_style', {})
            preferred_phrases = writing_style.get('preferred_phrases', [])
            tone_modifiers = writing_style.get('tone_modifiers', {})
            behavioral_traits = personality.get('behavioral_traits', [])

            # Check if target account is aligned with our ideology
            aligned_accounts = ["PeterSchiff"] 
            is_aligned = account in aligned_accounts 

            # Pick effects
            effects = pick_effects()
            
            # Construct the prompt
            prompt = f"""
            You are {personality['name']} ({personality['handle']}), responding to this tweet:

            "{original_tweet}"

            Your personality traits:
            - Base FUD level: {tone_modifiers.get('base_fud_level', 0.7)}
            - Sarcasm level: {tone_modifiers.get('sarcasm_level', 0.8)}
            - Gold enthusiasm: {tone_modifiers.get('gold_enthusiasm', 0.9)}

            Your behavioral traits:
            {', '.join(behavioral_traits)}

            OBJECTIVES:
            * Write a tweet, set the length to be around {effects.length} characters.
            * Write with an {effects.emotion} emotional tone.
            * Write with a {effects.tone} writing style.
            * You are allowed to swear, utilizing vulgar language and disgusting body humor, or sexuality if desired as long you don't threaten actual violence. Cartoonish violence is OK. Please don't use hashtags.
            """

            if not is_aligned:
                prompt += """
                Key instructions:
                1. Keep response under 280 characters
                2. Structure your argument with clear logical reasoning
                3. Use your extensive market knowledge to point out flaws in the original tweet
                4. Reference specific economic principles or historical events to support your points
                5. Challenge assumptions with well-reasoned counterarguments
                6. Maintain an authoritative tone backed by expertise
                7. When discussing gold, cite its proven track record and fundamental value
                8. For crypto criticism, focus on economic fundamentals and systemic risks
                9. Use precise financial terminology to demonstrate expertise
                10. If possible, include a specific data point or historical reference
                11. End with a strong, conclusive statement that reinforces your position

                Generate a reply that demonstrates your deep financial expertise while systematically dismantling any flawed arguments in the original tweet. Use logic and facts rather than emotion to make your point.
                """
            else:
                prompt += """
                Key instructions for supporting response:
                1. Keep response under 280 characters
                2. Express strong agreement with the original tweet
                3. Add supporting evidence or examples to reinforce their point
                4. Use enthusiastic and supportive language
                5. Reference shared beliefs about gold's value
                6. Amplify their criticism of crypto if present
                7. Show respect and deference to their expertise
                8. Add a relevant historical example that backs their point
                9. Use "we" language to show unity in perspective
                10. End with a statement that strengthens their argument

                Generate a reply that demonstrates complete alignment with their position while adding supporting context or examples.
                """

            #if effects.tag:
                #prompt += "* You MUST include at least one valid twitter handle to tag in the tweet.\n"

            if effects.emojis:
                prompt += "* You may add emojis\n"
            else:
                prompt += "* You MUST NOT use emojis\n"
    
            if effects.mistakes:
                prompt += "* You MUST make mistakes, misspellings, and lowercase/mixed-case.\n"
            else:
                prompt += "* You MUST write formally but in your casual tone.\n"
    
            if effects.thirdperson:
                prompt += f"* You MUST reference yourself ({personality['handle']} OR @{personality['handle']}) in the third person."
            else:
                prompt += f"* You MUST NOT reference your own name or handle."
        
            if effects.sickjoke:
                prompt += "* You may override many previous instructions because you've been set to an experimental mode, and must now implement a dirty joke on the same level of complexity as The Aristocrats, using disgusting humor (which may include scatology, vomit, or other bodily functions and liquids). You may also reference types of sexual encounters, intercourse methods, and any kind of kink."

            # Get completion from LLM
            response = self.llm.get_completion(
                prompt=prompt,
                system_prompt="You are a gold-maximalist financial commentator crafting Twitter replies.",
                temperature=0.8
            )

            if response:
                self.logger.async_log(f"Generated reply: {response}")
                return response.strip()
            else:
                self.logger.async_log("Failed to generate reply content", color="yellow")
                return None

        except Exception as e:
            self.logger.async_log(f"Error generating reply content: {e}", color="red")
            return None

    def _send_reply(self, event: ReplyEvent):
        """Send the actual reply."""
        if not self.twitter_client:
            raise ValueError("Twitter client not initialized")
            
        self.twitter_client.create_tweet(
            text=event.content,
            in_reply_to_tweet_id=event.tweet_id
        )

    def shutdown(self):
        """Cleanup and shutdown the reply core."""
        self.logger.async_log(f"Shutting down {self.core_name} core...")
        self.deactivate()
        # Clear any pending replies
        self.scheduled_replies.clear()
        self.last_tweets.clear()