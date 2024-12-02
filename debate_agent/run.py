#!/usr/bin/env python
from typing import List, Dict, Any
from colorama import Fore, Style, init
import json
import logging
from dotenv import load_dotenv
from litellm import completion
from naptha_sdk.utils import load_yaml
from naptha_sdk.schemas import AgentRunInput

from debate_agent.schemas import ACLMessage, ACLPerformative, InputSchema

logger = logging.getLogger(__name__)

load_dotenv()
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
    def __init__(self, llm_config):
        self.llm_config = llm_config

    def request(self, prompt: str) -> str:

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        if not isinstance(self.llm_config, dict):
            self.llm_config = self.llm_config.model_dump()
        result = completion(
            model=self.llm_config["model"],
            messages=messages,
            temperature=self.llm_config["temperature"],
            max_tokens=self.llm_config["max_tokens"],
            api_base=self.llm_config["api_base"],
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

def run(agent_run: AgentRunInput, *args, **kwargs):
    logger.info(f"Inputs: {agent_run.inputs}")

    if isinstance(agent_run.inputs, dict):
        agent_run.inputs = InputSchema(**agent_run.inputs)

    llm = LLM(agent_run.agent_deployment.agent_config.llm_config)

    if agent_run.inputs.agent_type == "debate":
        agent = Agent(agent_run.inputs.agent_name)
    elif agent_run.inputs.agent_type == "vera":
        agent = VeraAgent(agent_run.inputs.agent_name, llm)
    else:
        print("Agent type unknown")

    message = agent.generate_message(agent_run.inputs.conversation, llm)

    print(f"Message: {message}")
    return message.model_dump()

if __name__ == "__main__":
    agent_deployments_path = "debate_agent/configs/agent_deploymnets.json"
    with open(agent_deployments_path, "r") as f:
        agent_deployments = json.load(f)

    llm_configs_path = "debate_agent/configs/llm_configs.json"
    with open(llm_configs_path, "r") as f:
        llm_configs = json.load(f)

    agent_deployment = agent_deployments[0]
    llm_config_name = agent_deployment["agent_config"]["llm_config"]["config_name"]

    for config in llm_configs:
        if config["config_name"] == llm_config_name:
            llm_config = config
            break
    
    agent_deployment["agent_config"]["llm_config"] = llm_config

    initial_claim = "Tesla's price will exceed $250 in 2 weeks."
    initial_message = ACLMessage(
                performative=ACLPerformative.PROPOSE,
                sender="User",
                receiver="ALL",
                content=initial_claim,
                reply_with="msg1"
            )

    conversation = []
    conversation.append(initial_message.model_dump())

    inputs = {
        "agent_name": "debate_agent",
        "conversation": conversation,
        "agent_type": "debate"
    }

    agent_run_input = AgentRunInput(
        consumer_id="debate_agent",
        inputs=inputs,
        agent_deployment=agent_deployment
    )
    print(f"Agent run input: {agent_run_input}")

    message = run(agent_run_input)

    print("Message :", message)