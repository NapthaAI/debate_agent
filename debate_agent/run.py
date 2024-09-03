#!/usr/bin/env python
from typing import List, Dict, Any
from colorama import Fore, Style, init
import json
from dotenv import load_dotenv
from litellm import completion
from naptha_sdk.utils import get_logger, load_yaml
from debate_agent.schemas import ACLMessage, ACLPerformative, InputSchema

logger = get_logger(__name__)
load_dotenv()
# Initialize colorama
init(autoreset=True)

class Agent:
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initializing Agent {self.name}...")

    def generate_message(self, context: List[ACLMessage], llm: 'LLM') -> ACLMessage:
        # Create a prompt for the LLM
        context_summary = "\n".join([f"{msg.sender} ({msg.performative.value}): {msg.content}" for msg in context])
        acl_message_schema = ACLMessage.schema_json(indent=2)
        prompt = f"""Generate an ACL message for a debate within <acl> tags in JSON format. The current context is as follows:
    {context_summary}

    The JSON schema for the ACL message is:
    {acl_message_schema}

    Example:
    Respond in the format below with ACL JSON within <acl></acl> XML tags. Do not use ```json markdown block.
    <acl>
    {{
        "performative": "<PERFORMATIVE_TYPE>",
        "sender": "{self.name}",
        "receiver": "<RECEIVER_TYPE>",
        "content": "<CONTENT_MESSAGE>",
        "reply_with": "<REPLY_MESSAGE>",
        "language": "<LANGUAGE_TYPE>",
        "ontology": "<ONTOLOGY_TYPE>",
        "protocol": "<PROTOCOL_TYPE>",
        "conversation_id": "<CONVERSATION_ID>"
    }}
    </acl>
    """
        
        logger.info(f"Prompt {prompt}...")
        response = llm.request(prompt)
        logger.info(f"Response {response}...")

        message_data = parse_acl_response(response)
        if message_data:
            acl_msg = ACLMessage(**message_data)
            logger.info(f"ACL message: {acl_msg}")
            return acl_msg
        return None

class LLM:
    def __init__(self, api_key: str, model_url: str, cfg=None):
        self.api_key = api_key
        self.model_url = model_url
        self.cfg = cfg

    def request(self, prompt: str) -> str:

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        default_model_provider = self.cfg["models"]["default_model_provider"]

        result = completion(
            model=self.cfg["models"][default_model_provider]["model"],
            messages=messages,
            temperature=self.cfg["models"][default_model_provider]["temperature"],
            max_tokens=self.cfg["models"][default_model_provider]["max_tokens"],
            api_base=self.cfg["models"][default_model_provider]["api_base"],
        ).choices[0].message.content

        return result

class VeraAgent(Agent):
    def __init__(self, name: str, llm: LLM):
        super().__init__(name)
        self.llm = llm

    def generate_message(self, context: List[ACLMessage], llm: LLM) -> ACLMessage:
        debate_summary = "\n".join([f"{msg.sender}: {msg.content}" for msg in context])
        acl_message_schema = ACLMessage.schema_json(indent=2)
        prompt = f"""Based on the following debate about a market prediction, evaluate the arguments and evidence presented. Then, provide a judgment on the validity of the original prediction within <acl> tags in JSON format.

Debate summary:
{debate_summary}

The JSON schema for the ACL message is:
{acl_message_schema}

Example:
Respond in the format below with ACL JSON within XML tags. Do not use ```json markdown block.
<acl>
{{
  "performative": "<PERFORMATIVE_TYPE>",
  "sender": "{self.name}",
  "receiver": "<RECEIVER_TYPE>",
  "content": "<CONTENT_MESSAGE>",
  "reply_with": "<REPLY_MESSAGE>",
  "language": "<LANGUAGE_TYPE>",
  "ontology": "<ONTOLOGY_TYPE>",
  "protocol": "<PROTOCOL_TYPE>",
  "conversation_id": "<CONVERSATION_ID>"
}}
</acl>
"""
        response = llm.request(prompt)
        message_data = parse_acl_response(response)
        if message_data:
            acl_msg = ACLMessage(**message_data)
            logger.info(f"ACL message: {acl_msg}")
            return acl_msg
        return None


def parse_acl_response(response: str) -> Dict[str, Any]:
    try:
        # Locate <acl> tags
        start_idx = response.find("<acl>")
        end_idx = response.find("</acl>")
        
        # Check if both tags are present
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Invalid response format: Missing <acl> tags")
        
        # Extract JSON between <acl> and </acl>
        start_idx += len("<acl>")
        acl_json = response[start_idx:end_idx].strip()
        
        # Check if JSON is empty
        if not acl_json:
            raise ValueError("Empty JSON content in <acl> tags")

        # Clean up potential issues with the JSON string
        acl_json = acl_json.replace('\n', '').replace('\r', '')
        
        # Attempt to parse JSON data
        message_data = json.loads(acl_json)
        
        return message_data
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw Response: {response}")
        # Print the extracted JSON for debugging
        if 'acl_json' in locals():
            print(f"Extracted JSON: {acl_json}")
        return None

def run(inputs, worker_nodes=None, orchestrator_node=None, flow_run=None, cfg=None):
    logger.info(f"Inputs: {inputs}")

    api_key = None
    model_url = "https://api.openai.com/v1/chat/completions"
    llm = LLM(api_key, model_url, cfg)

    if inputs.agent_type == "debate":
        agent = Agent(inputs.agent_name)
    elif inputs.agent_type == "vera":
        agent = VeraAgent(inputs.agent_name, llm)
    else:
        print("Agent type unknown")

    message = agent.generate_message(inputs.conversation, llm)

    return message.json()

if __name__ == "__main__":
    cfg_path = "debate_agent/component.yaml"
    cfg = load_yaml(cfg_path)

    api_key = None
    model_url = "https://api.openai.com/v1/chat/completions"
    llm = LLM(api_key, model_url)

    initial_claim = "Tesla's price will exceed $250 in 2 weeks."
    initial_message = ACLMessage(
                performative=ACLPerformative.PROPOSE,
                sender="User",
                receiver="ALL",
                content=initial_claim,
                reply_with="msg1"
            )

    conversation = []
    conversation.append(initial_message)

    inputs = InputSchema(conversation=conversation, agent_type="debate")

    print(f"Conversation: {inputs.conversation}")

    message = run(inputs, cfg=cfg)

    print("Message :", message)