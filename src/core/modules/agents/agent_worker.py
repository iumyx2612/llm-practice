from llama_index.core.agent.types import Task, TaskStep
from llama_index.core.agent.react.step import (
    add_user_step_to_reasoning,
    ReActAgentWorker,
    TaskStepOutput
)
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
from .prompt_formatter import GoogleChatFormatter


class GeminiReActWorker(ReActAgentWorker):
    def __init__(self, *args, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._verbose = verbose
        self._react_chat_formatter = GoogleChatFormatter()

    def _run_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)
        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get() + task.extra_state["new_memory"].get_all(),
            current_reasoning=task.extra_state["current_reasoning"],
        )

        # send prompt
        chat_response = self._llm.complete(input_chat)
        # Convert from `CompletionResponse` to `ChatResponse` to fit with _process_actions
        chat_response = ChatResponse(
            message=ChatMessage(
                content=chat_response.text
            )
        )

        # given react prompt outputs, call tools or return response
        reasoning_steps, is_done = self._process_actions(
            task, tools, output=chat_response
        )
        task.extra_state["current_reasoning"].extend(reasoning_steps)
        agent_response = self._get_response(
            task.extra_state["current_reasoning"], task.extra_state["sources"]
        )
        if is_done:
            task.extra_state["new_memory"].put(
                ChatMessage(content=agent_response.response, role=MessageRole.ASSISTANT)
            )

        return self._get_task_step_response(agent_response, step, is_done)