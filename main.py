# -*- coding: utf-8 -*-
"""Run the SupportDesk agent interactively with in-memory conversation history.

Activate the project's virtual environment, then run: python main.py
The Chroma index must already exist. Questions can make model and embedding
API calls. Importing this file does not build or invoke the agent.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from openai import OpenAIError

from agent import build_agent


def extract_final_answer(state: dict) -> str:
    """Extract readable text from a completed assistant message."""
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("The agent returned no messages.")

    final_message = messages[-1]
    if not isinstance(final_message, AIMessage) or final_message.tool_calls:
        raise ValueError("The agent did not return a completed assistant answer.")

    content = final_message.content
    if isinstance(content, str):
        answer = content
    else:
        # Some models return content blocks rather than one string.
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        answer = "\n".join(parts)

    if not answer.strip():
        raise ValueError("The assistant answer contains no text.")
    return answer


def main() -> None:
    # Construct the agent once. This opens Chroma without rebuilding the index.
    agent = build_agent()
    history = []
    print("SupportDesk ready. Ask about your support tickets.")
    print("Type 'quit', 'exit', or 'q' to finish.")

    while True:
        try:
            question = input("\nYou: ").strip()
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if not question:
                continue

            # Send previous messages plus the new question. Create a new list
            # so a failed invocation does not add an incomplete turn to history.
            messages = history + [HumanMessage(content=question)]
            state = agent.invoke({"messages": messages})
            answer = extract_final_answer(state)

            # The result already contains earlier messages: replace history,
            # rather than appending the result and duplicating previous turns.
            # Keep tool calls and their responses together as part of history.
            history = state["messages"]
            print(f"Assistant: {answer}")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except GraphRecursionError:
            print("The agent reached its execution limit. Try a more specific question.")
        except OpenAIError as error:
            print(f"The API request failed: {error}")
        except ValueError as error:
            print(f"Could not read the answer: {error}")

    # History lives only in memory. Restarting the program starts a new chat.
    # Long conversations will eventually need deliberate trimming/summarization.


if __name__ == "__main__":
    main()
