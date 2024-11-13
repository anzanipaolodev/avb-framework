import os
import replicate
from openai import OpenAI
from typing import Dict, Any
from dotenv import load_dotenv

class LLMEngine:
    _instance = None
    
    def __init__(self):
        """Initialize LLM Engine with configuration from environment"""
        load_dotenv()
        
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model_version = None
        self.client = None
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_version = os.getenv("LLM_MODEL")
        elif self.provider == "replicate":
            self.client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
            self.model_version = os.getenv("REPLICATE_MODEL_VERSION")
        
        self.validate_configuration()
    
    @classmethod
    def get_instance(cls) -> 'LLMEngine':
        """Get singleton instance of LLMEngine"""
        if cls._instance is None:
            cls._instance = LLMEngine()
        return cls._instance
    
    def validate_configuration(self):
        """Validate the LLM configuration"""
        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("Required environment variable OPENAI_API_KEY is missing")
            try:
                available_models = [model.id for model in self.client.models.list().data]
                if self.model_version not in available_models:
                    raise ValueError(f"Model {self.model_version} is not available")
            except Exception as e:
                raise ValueError(f"Failed to validate OpenAI configuration: {str(e)}")
                
        elif self.provider == "replicate":
            if not os.getenv("REPLICATE_API_TOKEN"):
                raise ValueError("Required environment variable REPLICATE_API_TOKEN is missing")
            try:
                # Test API connection
                self.client.models.get(self.model_version.split(":")[0])
            except Exception as e:
                raise ValueError(f"Failed to validate Replicate configuration: {str(e)}")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def get_completion(self, 
                      prompt: str, 
                      system_prompt: str = None,
                      prompt_template: str = None,
                      temperature: float = 1.0,
                      max_tokens: int = 500) -> str:
        """
        Get completion from the configured LLM provider
        
        Args:
            prompt: The main prompt text
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            str: The generated completion text
        """
        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            completion = self.client.chat.completions.create(
                model=self.model_version,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return completion.choices[0].message.content
            
        elif self.provider == "replicate":
            output = replicate.run(
                self.model_version,
                input={
                    "system_prompt": system_prompt if system_prompt else "",
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 1,
                    "prompt_template": prompt_template if prompt_template else """
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    {system_prompt}<|eot_id|><|start_header_id|>instructions<|end_header_id|>
                    {prompt}<|eot_id|><|start_header_id|>response<|end_header_id|>
                    """,
                }
            )
            return ''.join([chunk for chunk in output])
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get current provider configuration info"""
        return {
            "provider": self.provider,
            "model": self.model_version
        } 